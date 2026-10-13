"""Explicit single-color wording is enforced independently of the LLM's SQL."""
import re
from query_engine import COLOR_NAMES

ALIASES={'gray':'grey','sky blue':'light blue'}
COLOR_PATTERN=re.compile(r'\b(?:'+ '|'.join(re.escape(c) for c in sorted(COLOR_NAMES|set(ALIASES),key=lambda c:(-len(c),c)))+r')\b',re.I)


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
