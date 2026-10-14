"""Explicit single-color wording is enforced independently of the LLM's SQL."""
import re
from query_engine import COLOR_NAMES, validate_sql
import sqlglot
from sqlglot import exp

ALIASES={'gray':'grey','sky blue':'light blue'}
COLOR_PATTERN=re.compile(r'\b(?:'+ '|'.join(re.escape(c) for c in sorted(COLOR_NAMES|set(ALIASES),key=lambda c:(-len(c),c)))+r')\b',re.I)


def search_preferences(query):
    """Keep occasion/comfort as ranking hints rather than unavailable hard fields."""
    text = re.sub(r'\b(shrt|shrit|officie|offce|uder)\b',
                  lambda m: {'shrt':'shirt','shrit':'shirt','officie':'office','offce':'office','uder':'under'}[m.group().lower()],
                  query, flags=re.I)
    preferences = {}
    for name, phrase in [('office', r'(?:office(?:\s+work)?|workplace|business\s+casual)'),
                         ('comfort', r'(?:comfortable|comfy|comfort)')]:
        pattern = re.compile(r'\b'+phrase+r'\b', re.I)
        matches = list(pattern.finditer(text))
        # Preserve negative requests for the LLM; never reverse their meaning.
        if matches and not any(re.search(r'\b(?:not|no|without|except|avoid|excluding)\s+(?:for\s+)?$',text[:m.start()],re.I) for m in matches):
            preferences[name] = True
            text = re.sub(r'\b(?:for|at|to)\s+(?:the\s+)?('+phrase+r')\b', ' ', text, flags=re.I)
            text = pattern.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip() or 'shirts', preferences


def preference_filter_sql(sql, query, columns):
    """Office does not imply an obligatory plain pattern."""
    normalized=validate_sql(sql,columns)
    if re.search(r'\b(?:plain|solid|pattern\w*|strip\w*|check\w*|plaid|dot\w*|floral|flower\w*|graphic\w*|print\w*|camouflage|camo|color[ -]block|colour[ -]block)\b',query,re.I):
        return normalized
    tree=sqlglot.parse_one(normalized,read='duckdb')
    for node in list(tree.find_all(exp.In,exp.EQ,exp.NEQ,exp.Like,exp.ILike,exp.Contains)):
        fields=list(node.find_all(exp.Column))
        if not fields or any(c.name!='pattern' for c in fields):continue
        ancestor=node.parent
        while ancestor is not None and not isinstance(ancestor,exp.Where):
            if not isinstance(ancestor,(exp.And,exp.Paren)):
                raise ValueError('The model added an ambiguous pattern restriction. Please retry your query.')
            ancestor=ancestor.parent
        node.replace(exp.Boolean(this=True))
    return tree.sql(dialect='duckdb')


def single_color_intent(query):
    text=query.lower()
    plain=bool(re.search(r'\b(?:plain|solid)\b',text))
    exclusive=bool(re.search(r'\b(?:only|single[- ]colou?red|monochrome)\b',text))
    if not (plain or exclusive) or re.search(r'\bnot\s+(?:plain|solid)\b',text):return None,False
    matches=list(COLOR_PATTERN.finditer(text))
    colors={ALIASES.get(m.group(),m.group()) for m in matches}
    if len(colors)!=1:return None,False
    for m in matches:
        if re.search(r'\b(?:not|except|exclude|excluding|without)\s*$',text[:m.start()]):
            return None,False
    return next(iter(colors)),plain


def dominant_color_intent(query):
    """A single positive shade means the shirt's predominant color."""
    text=query.lower()
    matches=list(COLOR_PATTERN.finditer(text))
    colors={ALIASES.get(m.group(),m.group()) for m in matches}
    if len(colors)!=1:return None
    for match in matches:
        if re.search(r'\b(?:not|except|exclude|excluding|without|no)\s*$',text[:match.start()]):return None
    return next(iter(colors))
