"""Application-owned constraints for explicit, supported request wording."""
import re
from query_engine import validate_sql
import sqlglot
from sqlglot import exp
from request_policy import COLOR_PATTERN, ALIASES
from product_enrichment import SIZE_RE, canonical_size

NEGATION=re.compile(r'\b(?:not|no|without|except|excluding|exclude|avoid)\s+(?:a\s+|any\s+|the\s+)?$',re.I)

def requested_sort(query):
    choices=[]
    for sort,pattern in [
        ('lowest_price',r'cheapest|lowest[ -]price|least expensive|price\s+(?:low to high|ascending)'),
        ('highest_rating',r'highest[ -]rated|highest[ -]rating|best[ -]rated|top[ -]rated'),
        ('most_reviewed',r'most[ -]reviewed|most popular|most ratings')]:
        for m in re.finditer(r'\b(?:'+pattern+r')\b',query,re.I):
            if not NEGATION.search(query[:m.start()]):choices.append(sort)
    if len(set(choices))>1:raise ValueError('Choose one sort order: cheapest, highest rated, or most reviewed.')
    return choices[0] if choices else 'recommended'

def literal(value):
    return exp.Literal.string(value).sql(dialect='duckdb')

def token(field,value,delimiter):
    return f"contains({literal(delimiter)} || lower(coalesce({field},'')) || {literal(delimiter)}, {literal(delimiter+value.lower()+delimiter)})"

def terms(query,matches,field,make):
    """Support conjunctions/alternatives, reject ambiguous mixed grouping."""
    matches=list(matches)
    if not matches:return None
    clauses=[];joins=[];negative=[]
    for i,m in enumerate(matches):
        neg=bool(NEGATION.search(query[:m.start()]))
        clause=make(m)
        clauses.append('NOT ('+clause+')' if neg else clause);negative.append(neg)
        if i:
            span=query[matches[i-1].end():m.start()]
            joins.append('OR' if re.search(r'\bor\b',span,re.I) else 'AND')
    if len(set(joins))>1 or ('OR' in joins and any(negative)):
        raise ValueError(f'Ambiguous {field} alternatives; use a simple AND or OR request.')
    join=joins[0] if joins else 'AND'
    return '('+f' {join} '.join(dict.fromkeys(clauses))+')'

def explicit_constraints(query):
    filters={}
    color=terms(query,COLOR_PATTERN.finditer(query),'color',
        lambda m:token('color',ALIASES.get(m.group().lower(),m.group().lower()),'|'))
    if color:filters['color']=color
    materials=r'cotton|polyester|linen|nylon|viscose|rayon|spandex|elastane|silk|wool|modal|bamboo'
    fabric=terms(query,re.finditer(r'\b(?:'+materials+r')\b',query,re.I),'fabric',lambda m:token('fabric',m.group().lower(),';'))
    if fabric:filters['fabric']=fabric
    # Bare one-letter sizes must be uppercase; lowercase sizes need explicit context.
    sizes=[m for m in SIZE_RE.finditer(query) if re.search(r'\bsize(?:s)?\s*[:=]?\s*$',query[:m.start()],re.I)
           or re.fullmatch(r'(?:XXS|XS|S|M|L|XL|XXL|[3-8]XL)(?: Tall)?',m.group())]
    size=terms(query,sizes,'size',lambda m:token('sizes',canonical_size(m.group()),'|'))
    if size:filters['sizes']=size
    sleeve=terms(query,re.finditer(r'\b(?:full|long|half|short)[ -]sleeve(?:d|s)?\b',query,re.I),'sleeve',
        lambda m:'sleeve = '+literal('full' if m.group().lower().startswith(('full','long')) else 'half'))
    if sleeve:filters['sleeve']=sleeve
    patterns={'plain':'plain','solid':'plain','checked':'checked_plaid','checkered':'checked_plaid','plaid':'checked_plaid',
        'floral':'floral','polka dot':'polka_dot','polka-dot':'polka_dot','camouflage':'camouflage',
        'horizontal stripes':'horizontal_stripes','horizontally striped':'horizontal_stripes',
        'vertical stripes':'vertical_stripes','vertically striped':'vertical_stripes',
        'diagonal stripes':'diagonal_stripes','diagonally striped':'diagonal_stripes',
        'color block':'color_block','colour block':'color_block','graphic print':'graphic_print'}
    regex=r'\b(?:'+'|'.join(re.escape(x) for x in sorted(patterns,key=len,reverse=True))+r'|striped|stripes)\b'
    def pattern_clause(m):
        value=patterns.get(m.group().lower())
        return 'pattern = '+literal(value) if value else "pattern IN ('horizontal_stripes','vertical_stripes','diagonal_stripes')"
    pattern=terms(query,re.finditer(regex,query,re.I),'pattern',pattern_clause)
    if pattern:filters['pattern']=pattern
    pocket=terms(query,re.finditer(r'\b(?:no|zero|0|one|single|1|two|double|2)\s+pockets?\b',query,re.I),'pocket',
        lambda m:'pocket = '+literal({'no':'no','zero':'no','0':'no','one':'single','single':'single','1':'single','two':'double','double':'double','2':'double'}[m.group().lower().split()[0]]))
    if pocket:filters['pocket']=pocket
    # 'Black or white, cotton or linen' is independent alternatives; 'black
    # cotton or white linen' pairs the fields. Do not invent cross combinations.
    spans=[]
    for field,regex in [('color',COLOR_PATTERN),('fabric',re.compile(r'\b(?:'+materials+r')\b',re.I)),
                        ('pattern',re.compile(regex,re.I)),('sizes',SIZE_RE),
                        ('sleeve',re.compile(r'\b(?:full|long|half|short)[ -]sleeve(?:d|s)?\b',re.I))]:
        if ' OR ' not in filters.get(field,''):continue
        found=list(regex.finditer(query))
        if len(found)>1:spans.append((found[0].start(),found[-1].end()))
    for i,(start,end) in enumerate(spans):
        if any(max(start,a)<min(end,b) for a,b in spans[i+1:]):
            raise ValueError('Paired alternatives across fields need separate searches; please split this request.')
    budget_re=r'\b(under|below|less than|up to|at most|over|above|more than|at least)\s*(₹|\$|rs\.?|inr|usd)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(rupees?|inr|rs\.?|usd|dollars?)?'
    prior_budget=None
    for m in re.finditer(budget_re,query,re.I):
        if re.search(r'\b(?:rating|rated|ratings|reviews|stars)\s*$',query[:m.start()],re.I):continue
        if NEGATION.search(query[:m.start()]):raise ValueError('Rephrase the budget as under, up to, above, or at least an amount.')
        if prior_budget and re.search(r'\bor\b',query[prior_budget.end():m.start()],re.I):
            raise ValueError('Alternative price ranges need separate searches; please split this request.')
        prior_budget=m
        currency=' '.join(x or '' for x in [m[2],m[4]]).lower()
        if any(x in currency for x in ['usd','dollar','$']):field='price_usd'
        else:field='price_inr_est'
        op={'under':'<','below':'<','less than':'<','up to':'<=','at most':'<=','over':'>','above':'>','more than':'>','at least':'>='}[m[1].lower()]
        amount=m[3].replace(',','')
        condition=f'{field} {op} {amount}'
        filters[field]=filters[field]+' AND '+condition if field in filters else condition
    return filters

def enforce_request_sql(sql,query,columns):
    """Repair controlled field predicates only after the SQL safety check.

    A mixed-field OR/NOT cannot be safely rewritten; reject instead of silently
    dropping part of the model's condition. Other explicit SQL conditions remain.
    """
    safe=validate_sql(sql,columns,normalize_predicates=False)
    filters=explicit_constraints(query)
    if not filters:return validate_sql(safe,columns),[]
    missing=set(filters)-set(columns)
    if missing:raise ValueError('Requested fields are unavailable: '+', '.join(sorted(missing)))
    tree=sqlglot.parse_one(safe,read='duckdb')
    def preserve(node):
        if isinstance(node,exp.Paren):return preserve(node.this)
        if isinstance(node,exp.And):
            left,right=preserve(node.this),preserve(node.expression)
            if isinstance(left,exp.Boolean) and left.this is True:return right
            if isinstance(right,exp.Boolean) and right.this is True:return left
            return exp.and_(left,right)
        fields={c.name for c in node.find_all(exp.Column)}
        if fields & set(filters):
            if not fields<=set(filters):raise ValueError('Model combined requested and unrelated filters ambiguously. Please retry.')
            return exp.Boolean(this=True)
        return node.copy()
    where=tree.args.get('where')
    kept=preserve(where.this) if where else exp.Boolean(this=True)
    required=[sqlglot.parse_one(condition,read='duckdb') for condition in filters.values()]
    clauses=required if isinstance(kept,exp.Boolean) and kept.this is True else [kept,*required]
    tree.set('where',exp.Where(this=exp.and_(*clauses)))
    return validate_sql(tree.sql(dialect='duckdb'),columns),list(filters)
