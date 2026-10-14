"""Validated filter SQL, restricted Parquet access and deterministic ranking."""
import json
from pathlib import Path
import sys
import threading
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent
if (ROOT / '.vendor').is_dir(): sys.path.insert(0, str(ROOT / '.vendor'))
import duckdb
import sqlglot
from sqlglot import exp
from product_enrichment import canonical_size
from catalog_quality import non_shirt_reason, palette_dominance

FILTER_COLUMNS = {'parent_asin','title','store','brand','sizes','color','color_main','pattern','sleeve',
    'style','fabric','fit_type','pocket','average_rating','rating_number','review_count',
    'price_usd','price_inr_est','price_source','fit_feedback','fit_review_n',
    'vision_status','color_status'}

COLOR_NAMES = set('black|charcoal|grey|silver|white|cream|beige|tan|khaki|brown|rust|yellow|mustard|gold|orange|coral|peach|red|burgundy|maroon|pink|hot pink|purple|lavender|plum|blue|light blue|denim|navy|indigo|teal|turquoise|green|dark green|lime|mint|olive|sage'.split('|'))
PATTERN_NAMES = set('plain|horizontal_stripes|vertical_stripes|diagonal_stripes|checked_plaid|polka_dot|floral|graphic_print|camouflage|color_block|other|unknown'.split('|'))


def exact_size_predicates(tree):
    """A size M filter must never match XL, XXL, or model measurements."""
    for node in list(tree.find_all(exp.Like,exp.ILike,exp.Contains,exp.EQ,exp.In)):
        field=node.this
        while isinstance(field,(exp.Lower,exp.Upper,exp.Trim,exp.Coalesce)): field=field.this
        if not isinstance(field,exp.Column) or field.name.lower()!='sizes': continue
        values=node.expressions if isinstance(node,exp.In) else [node.expression]
        clauses=[]
        for literal in values:
            if not isinstance(literal,exp.Literal) or not literal.is_string:
                raise ValueError('Size matching needs a literal size label')
            size=canonical_size(literal.this.strip('%').strip())
            if size is None: raise ValueError('Use a supported size label such as M, L, XL or XXL')
            clauses.append(sqlglot.parse_one("contains('|' || lower(coalesce("+field.sql(dialect='duckdb')+",'')) || '|', '|"+size.lower()+"|')",read='duckdb'))
        if not clauses or (isinstance(node,exp.In) and node.args.get('query')): raise ValueError('Size IN needs literal sizes')
        node.replace(exp.or_(*clauses))
    return tree


def canonical_pattern_predicates(tree):
    aliases = {'striped':['horizontal_stripes','vertical_stripes','diagonal_stripes'],
        'stripes':['horizontal_stripes','vertical_stripes','diagonal_stripes'],
        'stripe':['horizontal_stripes','vertical_stripes','diagonal_stripes'],
        'plaid':['checked_plaid'],'checked':['checked_plaid'],'checkered':['checked_plaid'],
        'checks':['checked_plaid'],'check':['checked_plaid'],'solid':['plain'],'solid_color':['plain']}
    for node in list(tree.find_all(exp.Like,exp.ILike,exp.Contains,exp.EQ,exp.In)):
        field=node.this
        while isinstance(field,(exp.Lower,exp.Upper,exp.Trim,exp.Coalesce)):field=field.this
        if not isinstance(field,exp.Column) or field.name.lower()!='pattern':continue
        literals=node.expressions if isinstance(node,exp.In) else [node.expression]
        if not literals or (isinstance(node,exp.In) and node.args.get('query')):
            raise ValueError('Pattern matching needs literal pattern names')
        names=[]
        for value in literals:
            if not isinstance(value,exp.Literal) or not value.is_string:
                raise ValueError('Pattern matching needs a literal pattern name')
            name=value.this.strip('%').strip().lower().replace(' ','_').replace('-','_')
            mapped=aliases.get(name,[name])
            if any(x not in PATTERN_NAMES for x in mapped):
                raise ValueError('Use a supported pattern name')
            names.extend(mapped)
        node.replace(exp.In(this=field.copy(),expressions=[exp.Literal.string(x) for x in dict.fromkeys(names)]))
    return tree


def exact_color_predicates(tree):
    """Normalize common model color filters to exact tokens, including mixed colors."""
    def color_column(field):
        while isinstance(field,(exp.Lower,exp.Upper,exp.Trim,exp.Coalesce)): field=field.this
        return field if isinstance(field,exp.Column) and field.name.lower()=='color' else None

    def token_predicate(field,literal):
        if not isinstance(literal,exp.Literal) or not literal.is_string:
            raise ValueError('Color matching needs a literal shade name')
        names=[x.strip().lower() for x in literal.this.strip('%').split('|')]
        names=[{'gray':'grey','sky blue':'light blue'}.get(x,x) for x in names]
        if names==['']: return exp.EQ(this=field.copy(),expression=literal.copy())
        if any(name not in COLOR_NAMES for name in names):
            raise ValueError('Use a complete supported shade name for color matching')
        clauses=[sqlglot.parse_one("contains('|' || lower(coalesce(" + field.sql(dialect='duckdb')
            + ",'')) || '|', '|" + name + "|')",read='duckdb') for name in names]
        return exp.and_(*clauses)

    for node in list(tree.find_all(exp.Like,exp.ILike,exp.Contains,exp.EQ,exp.In)):
        field=color_column(node.this)
        literal=node.expression
        if field is None and isinstance(node,exp.EQ):
            field=color_column(node.expression);literal=node.this
        if field is None: continue
        if isinstance(node,exp.In):
            if node.args.get('query') or not node.expressions:
                raise ValueError('Color IN needs a list of literal shade names')
            node.replace(exp.or_(*(token_predicate(field,x) for x in node.expressions)))
        else:node.replace(token_predicate(field,literal))
    return tree


def validate_sql(sql, columns, *, normalize_predicates=True):
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 8000:
        raise ValueError('SQL must be a nonempty, bounded string')
    try: statements = sqlglot.parse(sql, read='duckdb')
    except sqlglot.errors.ParseError as exc: raise ValueError('SQL cannot be parsed') from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError('Only one SELECT query is allowed')
    tree = statements[0]
    if normalize_predicates:
        tree = exact_size_predicates(canonical_pattern_predicates(exact_color_predicates(tree)))
    forbidden_args = ['with_', 'joins', 'group', 'having', 'qualify', 'order', 'limit',
                      'offset', 'distinct', 'into', 'windows', 'locks']
    if any(tree.args.get(k) for k in forbidden_args):
        raise ValueError('Use a single catalog filter; the application handles ranking and limits')
    if len(tree.expressions) != 1 or not isinstance(tree.expressions[0], exp.Column) or tree.expressions[0].name.lower() != 'parent_asin':
        raise ValueError('SELECT must return only parent_asin')
    tables = list(tree.find_all(exp.Table))
    if len(tables) != 1 or not isinstance(tables[0].this, exp.Identifier) or tables[0].name.lower() != 'shirts' or tables[0].db or tables[0].catalog:
        raise ValueError('Only the shirts table is available')
    from_node = tree.args.get('from_')
    if from_node is None or not isinstance(from_node.this, exp.Table):
        raise ValueError('FROM shirts is required')
    aliases = {'', 'shirts', tables[0].alias.lower()}
    allowed_functions = {'CONTAINS','LOWER','UPPER','COALESCE','ABS','ROUND','TRIM'}
    allowed_nodes = {'Select','From','Table','TableAlias','Identifier','Column','Where',
        'Literal','Null','Boolean','Paren','And','Or','Not','EQ','NEQ','GT','GTE','LT','LTE',
        'Is','In','Between','Like','ILike','DPipe','Neg','Add','Sub','Mul','Div'}
    nodes = list(tree.walk())
    if len(nodes) > 180: raise ValueError('SQL expression is too complex')
    for node in nodes:
        if type(node).__name__ in allowed_nodes:
            pass
        elif isinstance(node, exp.Func):
            if node.sql_name() not in allowed_functions: raise ValueError('Unsupported SQL function')
        else:
            raise ValueError('Unsupported SQL expression: ' + type(node).__name__)
        if isinstance(node, exp.Column):
            if node.name not in columns or node.table.lower() not in aliases or node.db or node.catalog:
                raise ValueError('Unknown or unavailable catalog column: ' + node.name)
    return tree.sql(dialect='duckdb')


class Catalog:
    def __init__(self, path=None):
        if path is None:
            path=ROOT/'data/shirts_catalog.parquet'
            if not path.is_file():path=ROOT/'data/shirts_master.parquet'
        self.path = Path(path).resolve()
        self.options_path=self.path.parent/'shirt_size_options.parquet' if self.path.name=='shirts_catalog.parquet' else None
        if self.options_path is not None and not self.options_path.is_file():
            raise FileNotFoundError('The compact catalog requires its recorded size options file.')
        self.compact=bool(self.options_path and self.options_path.is_file())
        self._base_columns=[];self._option_columns=[]
        if self.compact:
            import pyarrow.parquet as pq
            self._base_columns=pq.read_schema(self.path).names
            self._option_columns=pq.read_schema(self.options_path).names
        self.details_path = self.path.parent / 'product_details.parquet'
        if not self.path.is_file(): raise FileNotFoundError('Master Parquet is missing')
        self._quality=None
        with self.connect() as con:
            self.schema = [(r[0],r[1]) for r in con.execute('DESCRIBE shirts').fetchall()]
            self.columns = set(c for c, _ in self.schema)
            self.prior_rating = float(con.execute('SELECT coalesce(avg(average_rating) FILTER (WHERE average_rating > 0), 4) FROM shirts').fetchone()[0])
            quality_fields=[c for c in ['parent_asin','title','color','color_candidates'] if c in self.columns]
            quality_input=con.execute('SELECT '+','.join('"'+c+'"' for c in quality_fields)+' FROM shirts').fetchdf()
        quality=[]
        for row in quality_input.to_dict('records'):
            reason=non_shirt_reason(row.get('title'))
            main,share=palette_dominance(row.get('color_candidates'))
            fallback=False
            if 'color_candidates' not in self.columns:
                # Legacy catalogs may only provide one label: do not fabricate
                # a pixel percentage. Mixed tags cannot prove predominance.
                label=str(row.get('color') or '').strip().lower()
                if label in COLOR_NAMES:main=label;fallback=True
            quality.append({'parent_asin':row['parent_asin'],'eligible':not bool(reason),
                            'exclusion_reason':reason,'main_color':main,'main_color_share':share,'label_fallback':fallback})
        self._quality=pd.DataFrame(quality)
        self.filter_columns = self.columns & FILTER_COLUMNS
        self._options = None

    def connect(self):
        con = duckdb.connect(config={'memory_limit':'512MB', 'threads':2,
            'autoinstall_known_extensions':False, 'autoload_known_extensions':False})
        escaped = self.path.as_posix().replace("'", "''")
        # Set allowed paths BEFORE disabling external access (DuckDB requires this order).
        allowed=[escaped]
        if self.details_path.is_file(): allowed.append(self.details_path.as_posix().replace("'","''"))
        if self.compact:allowed.append(self.options_path.as_posix().replace("'","''"))
        con.execute("SET allowed_paths = [" + ','.join("'"+p+"'" for p in allowed) + "]")
        con.execute('SET enable_external_access = false')
        con.execute("CREATE VIEW catalog_products AS SELECT * FROM read_parquet('"+escaped+"')")
        if self.compact:
            option_path=self.options_path.as_posix().replace("'","''")
            con.execute("CREATE VIEW size_options AS SELECT * FROM read_parquet('"+option_path+"')")
            projection=','.join(('v' if name in self._option_columns else 'c')+'."'+name+'" AS "'+name+'"' for name in self._base_columns)
            con.execute('CREATE VIEW shirts AS SELECT '+projection+', c.parent_asin AS catalog_product_id FROM catalog_products c JOIN size_options v ON v.catalog_product_id=c.parent_asin')
        else:con.execute('CREATE VIEW shirts AS SELECT * FROM catalog_products')
        if self._quality is not None:con.register('catalog_quality',self._quality)
        if self.details_path.is_file():
            con.execute("CREATE VIEW product_details AS SELECT * FROM read_parquet('"+allowed[1]+"')")
        con.execute('SET lock_configuration = true')
        return con

    def overview(self):
        with self.connect() as con:
            base = con.execute("SELECT count(*) products, count(*) FILTER (WHERE color <> '') colors_available, count(*) FILTER (WHERE pattern <> 'unknown') patterns_available FROM shirts").fetchdf().iloc[0].to_dict()
            listing_count=int(base['products'])
            if self.compact:
                base=con.execute("SELECT count(*) products, count(*) FILTER (WHERE color <> '') colors_available, count(*) FILTER (WHERE pattern <> 'unknown') patterns_available FROM catalog_products").fetchdf().iloc[0].to_dict()
            group_key="CASE WHEN trim(coalesce(s.image_url,'')) <> '' THEN 'url:' || trim(s.image_url) ELSE 'id:' || s.parent_asin END" if 'image_url' in self.columns else "s.parent_asin"
            image_count=con.execute('SELECT count(DISTINCT '+group_key+') FROM shirts s').fetchone()[0]
            searchable_count=con.execute('SELECT count(DISTINCT '+group_key+') FROM shirts s JOIN catalog_quality q USING(parent_asin) WHERE q.eligible').fetchone()[0]
            vocab = {}
            for col in ['pattern','sleeve','pocket','vision_status']:
                if col in self.columns:
                    vocab[col] = [r[0] for r in con.execute(f'SELECT DISTINCT "{col}" FROM shirts WHERE "{col}" IS NOT NULL ORDER BY 1').fetchall()]
            vocab['color'] = [r[0] for r in con.execute("SELECT DISTINCT unnest(string_split(color, '|')) AS color_name FROM shirts WHERE color <> '' ORDER BY 1").fetchall()]
            if 'sizes' in self.columns:
                vocab['sizes'] = [r[0] for r in con.execute("SELECT DISTINCT unnest(string_split(sizes, '|')) AS size FROM shirts WHERE sizes <> '' ORDER BY 1").fetchall()]
        return {**{k:int(v) for k,v in base.items()},
                'recorded_listing_options':listing_count,'compact_catalog':self.compact,
                'searchable_products':int(self._quality.eligible.sum()),
                'unique_image_products':int(image_count),'searchable_image_products':int(searchable_count),
                'excluded_non_shirt_products':int((~self._quality.eligible).sum()), 'vocabulary':vocab,
                'columns':[{'name':c,'type':t} for c,t in self.schema if c in self.filter_columns]}

    def product(self, parent_asin):
        with self.connect() as con:
            rows=json.loads(con.execute('SELECT * FROM shirts WHERE parent_asin = ?', [parent_asin]).fetchdf().to_json(orient='records'))
            if not rows: return None
            product=rows[0]
            variants=[]
            if 'image_url' in self.columns and product.get('image_url'):
                fields=[c for c in ['parent_asin','title','brand','store','sizes','image_url','price_inr_est','price_source','color','sleeve'] if c in self.columns]
                brand_col='brand' if 'brand' in self.columns else 'store' if 'store' in self.columns else None
                clause=' AND lower(trim(coalesce(s."'+brand_col+'",\'\'))) = ?' if brand_col else ''
                params=[product['image_url'].strip()]
                if brand_col:params.append(str(product.get(brand_col) or '').strip().lower())
                variants=json.loads(con.execute('SELECT '+','.join('s."'+c+'"' for c in fields)+
                    ' FROM shirts s JOIN catalog_quality q USING(parent_asin) WHERE q.eligible AND trim(s.image_url)=?'+clause+
                    ' ORDER BY s.parent_asin',params).fetchdf().to_json(orient='records'))
            detail_rows=[]
            if self.details_path.is_file():
                detail_rows=json.loads(con.execute('SELECT * FROM product_details WHERE parent_asin = ?', [parent_asin]).fetchdf().to_json(orient='records'))
        details=detail_rows[0] if detail_rows else {}
        for source,target,default in [('features_json','features',[]),('description_json','description',[]),
                                      ('details_json','listing_details',{}),('image_urls_json','listing_images',[])]:
            product[target]=json.loads(details.get(source) or json.dumps(default))
        product['metadata_year']=details.get('metadata_year',2023)
        product['listing_title']=details.get('listing_title',product.get('title'))
        product['variants']=variants
        product['variant_group_basis']='Same recorded brand/store and primary image URL; listing relationships are inferred, not confirmed Amazon variant families.'
        return product

    def filter_options(self):
        if self._options is not None: return self._options
        options={'brands':[], 'colors':[], 'patterns':[], 'sizes':[], 'sleeves':[], 'fabrics':[], 'fits':[], 'pockets':[]}
        with self.connect() as con:
            if 'brand' in self.columns:
                grouped={}
                for name,count in con.execute("SELECT trim(brand),count(*) FROM shirts WHERE trim(coalesce(brand,'')) <> '' GROUP BY 1").fetchall():
                    key=name.lower()
                    item=grouped.setdefault(key,{'name':name,'products':0,'preferred_count':0})
                    item['products']+=count
                    if count>item['preferred_count']:item['name']=name;item['preferred_count']=count
                options['brands']=[{'name':v['name'],'products':v['products']} for v in sorted(grouped.values(),key=lambda v:v['name'].casefold())]
            for output,column,delimiter in [('colors','color','|'),('patterns','pattern',None),('sizes','sizes','|'),
                    ('sleeves','sleeve',None),('fabrics','fabric',';'),('fits','fit_type',None),('pockets','pocket',None)]:
                if column not in self.columns: continue
                expression=f"unnest(string_split(\"{column}\",'{delimiter}'))" if delimiter else f'"{column}"'
                values=con.execute(f'SELECT DISTINCT {expression} FROM shirts WHERE "{column}" IS NOT NULL').fetchall()
                options[output]=sorted({r[0].strip() for r in values if r[0] and r[0].strip() and r[0]!='unknown'},key=str.casefold)
        self._options=options
        return options

    def validate_selected_filters(self, filters):
        if not isinstance(filters,dict):raise ValueError('Selected filters must be an object')
        options=self.filter_options()
        normalized={}
        if filters.get('brands'):
            brands=filters['brands']
            if not isinstance(brands,list) or len(brands)>50:raise ValueError('Select up to 50 brands')
            lookup={b['name'].lower():b['name'] for b in options['brands']}
            if any(not isinstance(b,str) or b.strip().lower() not in lookup for b in brands):
                raise ValueError('Select a brand from the catalog list')
            normalized['brands']=list(dict.fromkeys(lookup[b.strip().lower()] for b in brands))
        for field,choice in [('color','colors'),('pattern','patterns'),('size','sizes'),('sleeve','sleeves'),
                             ('fabric','fabrics'),('fit','fits'),('pocket','pockets')]:
            if filters.get(field) in (None,''):continue
            if filters[field] not in options[choice]:raise ValueError('Select a catalog value for '+field)
            normalized[field]=filters[field]
        if filters.get('max_price') is not None:
            amount=filters['max_price']
            if isinstance(amount,bool) or not isinstance(amount,(int,float)) or not math.isfinite(amount) or not 0<amount<=10000000:
                raise ValueError('Maximum price must be a positive INR amount')
            normalized['max_price']=amount
        if set(filters)-{'brands','color','pattern','size','sleeve','fabric','fit','pocket','max_price'}:
            raise ValueError('Unknown selected filter')
        return normalized

    def search(self, sql, limit=12, include_uncertain=True, sort='recommended', only_color=None, require_plain=False, selected_filters=None, preferences=None, dominant_color=None):
        if not 1 <= limit <= 50: raise ValueError('Result limit must be between 1 and 50')
        orders = {'recommended':"CASE WHEN s.price_source LIKE 'exact%' THEN 0 ELSE 1 END, rank_score DESC, s.rating_number DESC",
                  'lowest_price':'s.price_inr_est ASC NULLS LAST, rank_score DESC',
                  'highest_rating':'s.average_rating DESC NULLS LAST, s.rating_number DESC',
                  'most_reviewed':'s.rating_number DESC, rank_score DESC'}
        if sort not in orders: raise ValueError('Unsupported result sort')
        preferences=preferences or {}
        if any(k not in {'office','comfort'} or v is not True for k,v in preferences.items()):
            raise ValueError('Unsupported search preference')
        selected=self.validate_selected_filters(selected_filters or {})
        if selected.get('color') and dominant_color is None:dominant_color=selected['color']
        if dominant_color is not None and dominant_color not in COLOR_NAMES:raise ValueError('Unsupported dominant color')
        if selected.get('color') and selected.get('pattern')=='plain' and only_color is None:
            only_color=selected['color'];require_plain=True
        normalized = validate_sql(sql, self.filter_columns)
        if only_color is not None:
            if only_color not in COLOR_NAMES:raise ValueError('Unsupported exclusive color')
            tree=sqlglot.parse_one(normalized,read='duckdb')
            # Add after broad multi-color normalization: this equality deliberately
            # requires the entire stored color list to contain just this one name.
            strict=sqlglot.parse_one("lower(trim(coalesce(color,''))) = '"+only_color+"'",read='duckdb')
            if require_plain:strict=exp.and_(strict,sqlglot.parse_one("pattern = 'plain'",read='duckdb'))
            tree=tree.where(strict,append=True)
            normalized=tree.sql(dialect='duckdb')
        if selected:
            tree=sqlglot.parse_one(normalized,read='duckdb')
            clauses=[]
            if selected.get('brands'):
                field=sqlglot.parse_one("lower(trim(coalesce(brand,'')))",read='duckdb')
                clauses.append(exp.In(this=field,expressions=[exp.Literal.string(b.lower()) for b in selected['brands']]))
            for field,column,delimiter in [('color','color','|'),('size','sizes','|'),('fabric','fabric',';')]:
                if field in selected:
                    haystack=sqlglot.parse_one(f"'{delimiter}' || lower(coalesce({column},'')) || '{delimiter}'",read='duckdb')
                    clauses.append(exp.Contains(this=haystack,expression=exp.Literal.string(delimiter+selected[field].lower()+delimiter)))
            for field,column in [('pattern','pattern'),('sleeve','sleeve'),('fit','fit_type'),('pocket','pocket')]:
                if field in selected:clauses.append(exp.EQ(this=exp.column(column),expression=exp.Literal.string(selected[field])))
            if 'max_price' in selected:clauses.append(exp.LTE(this=exp.column('price_inr_est'),expression=exp.Literal.number(selected['max_price'])))
            tree=tree.where(exp.and_(*clauses),append=True)
            normalized=tree.sql(dialect='duckdb')
        extra = '' if include_uncertain else " AND s.vision_status = 'ok'"
        extra += ' AND q.eligible'
        if dominant_color:
            # Application-owned evidence filter runs independently of model SQL.
            extra += " AND q.main_color = '"+dominant_color+"' AND (q.main_color_share >= 0.5 OR q.label_fallback)"
        matched = f'WITH matched AS ({normalized}) '
        fields = ['parent_asin','title','store','brand','sizes','image_url','color','pattern','sleeve','sleeve_source','sleeve_evidence',
            'fabric','fit_type','pocket','color_main','price_usd','price_inr_est','price_source',
            'average_rating','rating_number','review_count','vision_status','color_source','pattern_source']
        fields = [c for c in fields if c in self.columns]
        projection = ', '.join('s."' + c + '"' for c in fields)
        projection += ', q.main_color AS dominant_color, q.main_color_share AS dominant_color_share'
        if dominant_color:projection += ', q.main_color_share AS requested_color_share'
        # These expressions are application-owned, never generated by the LLM.
        # Rank before LIMIT so evidence on any matching listing can affect results.
        text_parts=[f"coalesce(s.\"{c}\",'')" for c in ['title','style','fabric','fit_type'] if c in self.columns]
        details_join=''
        if preferences and self.details_path.is_file():
            details_join=' LEFT JOIN product_details d ON s.parent_asin = d.parent_asin'
            text_parts += ["coalesce(d.features_json,'')", "coalesce(d.description_json,'')"]
        evidence_text="lower(concat_ws(' ',"+','.join(text_parts)+"))"
        def signal(words, weight):
            return f"CASE WHEN regexp_matches({evidence_text}, '(^|[^a-z])({words})([^a-z]|$)') THEN {weight} ELSE 0 END"
        scores=[]
        if preferences.get('office'):
            scores += [signal('office|formal|dress|business|oxford',4),signal('button[ -]down|button[ -]up|collared',2),
                       signal('hawaiian|beach|camouflage|graphic|costume|pajama|pyjama',-3)]
            if 'pattern' in self.columns: scores.append("CASE WHEN s.pattern = 'plain' THEN 1 ELSE 0 END")
        if preferences.get('comfort'):
            scores += [signal('comfortable|comfort|comfy|breathable|soft|stretch|lightweight',2),
                       signal('cotton|linen',1),signal('regular[ -]fit|relaxed[ -]fit|loose[ -]fit',1)]
        preference_score=' + '.join(scores) or '0'
        order=orders[sort]
        if dominant_color and sort=='recommended':order='q.main_color_share DESC NULLS LAST, '+order
        if preferences and sort=='recommended': order='preference_score DESC, '+order
        rank = '(coalesce(s.average_rating,0) * greatest(coalesce(s.rating_number,0),0) + 20 * ?) / (greatest(coalesce(s.rating_number,0),0) + 20)'
        predicate = 'WHERE s.parent_asin IN (SELECT parent_asin FROM matched)' + extra
        with self.connect() as con:
            timer = threading.Timer(8., con.interrupt); timer.daemon = True; timer.start()
            try:
                quality_join=' JOIN catalog_quality q ON s.parent_asin = q.parent_asin'
                group_key="CASE WHEN trim(coalesce(s.image_url,'')) <> '' THEN 'url:' || trim(s.image_url) ELSE 'id:' || s.parent_asin END" if 'image_url' in self.columns else 's.parent_asin'
                count = con.execute(matched + 'SELECT count(DISTINCT '+group_key+') FROM shirts s '+quality_join+' ' + predicate).fetchone()[0]
                # Rank matching listings first, retain one representative per URL,
                # then limit cards. Size/brand/budget filters are never relaxed.
                grouped=matched+f', scored AS (SELECT {projection}, {rank} AS rank_score, ({preference_score}) AS preference_score, '+group_key+' AS display_group FROM shirts s '+quality_join+details_join+' '+predicate+') '
                outer_order=order.replace('s.','').replace('q.main_color_share','dominant_color_share')
                frame=con.execute(grouped+'SELECT * EXCLUDE (display_group, image_row) FROM (SELECT *, '
                    'count(*) OVER (PARTITION BY display_group) AS matching_listing_count, '
                    'row_number() OVER (PARTITION BY display_group ORDER BY '+outer_order+', parent_asin) AS image_row FROM scored) '
                    'WHERE image_row=1 ORDER BY '+outer_order+', parent_asin LIMIT ?',[self.prior_rating,limit]).fetchdf()
            finally: timer.cancel()
        products = json.loads(frame.to_json(orient='records'))
        return {'sql':normalized, 'total_matches':int(count),'products':products,'only_color':only_color,'selected_filters':selected,
                'sort':sort, 'preferences':preferences, 'requested_dominant_color':dominant_color,
                'minimum_color_share':0.5 if dominant_color else None,
                'excluded_non_shirt_products':int((~self._quality.eligible).sum()), 'ranking':
                    ('Predominant requested shade required (at least 50% of saved shirt pixels); stronger color share ranks first. ' if dominant_color else '')+
                    ('Office/comfort listing evidence first, then price provenance and Bayesian rating. Preferences do not exclude matches. ' if preferences and sort=='recommended' else '')+
                    {'recommended':'Exact-price tier first; within each tier, Bayesian average rating with a 20-review prior.',
                    'lowest_price':'Lowest catalog INR price first, including labeled estimates.',
                    'highest_rating':'Highest catalog rating first; rating count breaks ties.',
                    'most_reviewed':'Highest number of catalog ratings first.'}[sort]}
