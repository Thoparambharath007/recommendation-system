"""Validated filter SQL, restricted Parquet access and deterministic ranking."""
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parent
if (ROOT / '.vendor').is_dir(): sys.path.insert(0, str(ROOT / '.vendor'))
import duckdb
import sqlglot
from sqlglot import exp

FILTER_COLUMNS = {'parent_asin','title','store','color','color_main','pattern','sleeve',
    'style','fabric','fit_type','pocket','average_rating','rating_number','review_count',
    'price_usd','price_inr_est','price_source','fit_feedback','fit_review_n',
    'vision_status','color_status'}

COLOR_NAMES = set('black|charcoal|grey|silver|white|cream|beige|tan|khaki|brown|rust|yellow|mustard|gold|orange|coral|peach|red|burgundy|maroon|pink|hot pink|purple|lavender|plum|blue|light blue|denim|navy|indigo|teal|turquoise|green|dark green|lime|mint|olive|sage'.split('|'))
PATTERN_NAMES = set('plain|horizontal_stripes|vertical_stripes|diagonal_stripes|checked_plaid|polka_dot|floral|graphic_print|camouflage|color_block|other|unknown'.split('|'))


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


def validate_sql(sql, columns):
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 8000:
        raise ValueError('SQL must be a nonempty, bounded string')
    try: statements = sqlglot.parse(sql, read='duckdb')
    except sqlglot.errors.ParseError as exc: raise ValueError('SQL cannot be parsed') from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError('Only one SELECT query is allowed')
    tree = canonical_pattern_predicates(exact_color_predicates(statements[0]))
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
    def __init__(self, path=ROOT / 'data/shirts_master.parquet'):
        self.path = Path(path).resolve()
        if not self.path.is_file(): raise FileNotFoundError('Master Parquet is missing')
        with self.connect() as con:
            self.schema = [(r[0],r[1]) for r in con.execute('DESCRIBE shirts').fetchall()]
            self.columns = set(c for c, _ in self.schema)
            self.prior_rating = float(con.execute('SELECT coalesce(avg(average_rating) FILTER (WHERE average_rating > 0), 4) FROM shirts').fetchone()[0])
        self.filter_columns = self.columns & FILTER_COLUMNS

    def connect(self):
        con = duckdb.connect(config={'memory_limit':'512MB', 'threads':2,
            'autoinstall_known_extensions':False, 'autoload_known_extensions':False})
        escaped = self.path.as_posix().replace("'", "''")
        # Set allowed paths BEFORE disabling external access (DuckDB requires this order).
        con.execute("SET allowed_paths = ['" + escaped + "']")
        con.execute('SET enable_external_access = false')
        con.execute("CREATE VIEW shirts AS SELECT * FROM read_parquet('" + escaped + "')")
        con.execute('SET lock_configuration = true')
        return con

    def overview(self):
        with self.connect() as con:
            base = con.execute("SELECT count(*) products, count(*) FILTER (WHERE color <> '') colors_available, count(*) FILTER (WHERE pattern <> 'unknown') patterns_available FROM shirts").fetchdf().iloc[0].to_dict()
            vocab = {}
            for col in ['pattern','sleeve','pocket','vision_status']:
                if col in self.columns:
                    vocab[col] = [r[0] for r in con.execute(f'SELECT DISTINCT "{col}" FROM shirts WHERE "{col}" IS NOT NULL ORDER BY 1').fetchall()]
            vocab['color'] = [r[0] for r in con.execute("SELECT DISTINCT unnest(string_split(color, '|')) AS color_name FROM shirts WHERE color <> '' ORDER BY 1").fetchall()]
        return {**{k:int(v) for k,v in base.items()}, 'vocabulary':vocab,
                'columns':[{'name':c,'type':t} for c,t in self.schema if c in self.filter_columns]}

    def search(self, sql, limit=12, include_uncertain=True, sort='recommended', only_color=None, require_plain=False):
        if not 1 <= limit <= 50: raise ValueError('Result limit must be between 1 and 50')
        orders = {'recommended':"CASE WHEN s.price_source LIKE 'exact%' THEN 0 ELSE 1 END, rank_score DESC, s.rating_number DESC",
                  'lowest_price':'s.price_inr_est ASC NULLS LAST, rank_score DESC',
                  'highest_rating':'s.average_rating DESC NULLS LAST, s.rating_number DESC',
                  'most_reviewed':'s.rating_number DESC, rank_score DESC'}
        if sort not in orders: raise ValueError('Unsupported result sort')
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
        extra = '' if include_uncertain else " AND s.vision_status = 'ok'"
        matched = f'WITH matched AS ({normalized}) '
        fields = ['parent_asin','title','store','image_url','color','pattern','sleeve','sleeve_source','sleeve_evidence',
            'fabric','fit_type','pocket','price_usd','price_inr_est','price_source',
            'average_rating','rating_number','review_count','vision_status','color_source','pattern_source']
        fields = [c for c in fields if c in self.columns]
        projection = ', '.join('s."' + c + '"' for c in fields)
        rank = '(coalesce(s.average_rating,0) * greatest(coalesce(s.rating_number,0),0) + 20 * ?) / (greatest(coalesce(s.rating_number,0),0) + 20)'
        predicate = 'WHERE s.parent_asin IN (SELECT parent_asin FROM matched)' + extra
        with self.connect() as con:
            timer = threading.Timer(8., con.interrupt); timer.daemon = True; timer.start()
            try:
                count = con.execute(matched + 'SELECT count(*) FROM shirts s ' + predicate).fetchone()[0]
                frame = con.execute(matched + f'SELECT {projection}, {rank} AS rank_score FROM shirts s '
                    + predicate + ' ORDER BY ' + orders[sort] + ', s.parent_asin LIMIT ?', [self.prior_rating,limit]).fetchdf()
            finally: timer.cancel()
        products = json.loads(frame.to_json(orient='records'))
        return {'sql':normalized, 'total_matches':int(count),'products':products,'only_color':only_color,
                'sort':sort, 'ranking':{'recommended':'Exact-price tier first; within each tier, Bayesian average rating with a 20-review prior.',
                    'lowest_price':'Lowest catalog INR price first, including labeled estimates.',
                    'highest_rating':'Highest catalog rating first; rating count breaks ties.',
                    'most_reviewed':'Highest number of catalog ratings first.'}[sort]}
