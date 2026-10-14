"""Local natural-language apparel search over the merged Parquet catalog."""
from functools import lru_cache
from pathlib import Path
import time
import re
from typing import Annotated

import duckdb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=False)
from query_engine import Catalog
from llm_sql import SQLGenerator, LLMError
from request_policy import single_color_intent, search_preferences, preference_filter_sql, dominant_color_intent
from request_guard import enforce_request_sql, requested_sort
from query_relevance import clarification_reason, SUGGESTIONS

app = FastAPI(title='Shirt catalog: LLM + DuckDB', docs_url='/docs')
app.mount('/assets', StaticFiles(directory=ROOT/'web'), name='assets')


@lru_cache(maxsize=1)
def catalog(): return Catalog()


@lru_cache(maxsize=1)
def metadata(): return catalog().overview()


class SelectedFilters(BaseModel):
    model_config = ConfigDict(extra='forbid')
    brands: list[Annotated[str,Field(min_length=1,max_length=2000)]] = Field(default_factory=list,max_length=50)
    color: str | None = None
    pattern: str | None = None
    size: str | None = None
    sleeve: str | None = None
    fabric: str | None = None
    fit: str | None = None
    pocket: str | None = None
    max_price: float | None = Field(default=None,gt=0,le=10000000,allow_inf_nan=False)


class SearchRequest(BaseModel):
    query: str = Field(default='', max_length=1000)
    limit: int = Field(default=12, ge=1, le=50)
    include_uncertain: bool = True
    filters: SelectedFilters = Field(default_factory=SelectedFilters)


@app.get('/')
def index(): return FileResponse(ROOT / 'web/index.html')


@app.get('/product/{parent_asin}')
def product_page(parent_asin: str = PathParam(pattern=r'^[A-Z0-9]{10}$')):
    return FileResponse(ROOT/'web/product.html')


@app.get('/api/health')
def health():
    generator = SQLGenerator()
    return {'ready':True, **metadata(), 'llm_configured':bool(generator.api_key), 'model':generator.model}


@app.get('/api/browse')
def browse(limit: int = Query(default=12,ge=1,le=50), include_uncertain: bool = True):
    return {**catalog().search('SELECT parent_asin FROM shirts',limit=limit,include_uncertain=include_uncertain),
            'explanation':'Browse the catalog, ranked by price provenance and rating evidence.',
            'unsupported':[], 'mode':'catalog_browse'}


@app.get('/api/filters')
def filter_options(): return catalog().filter_options()


@app.get('/api/products/{parent_asin}')
def product_details(parent_asin: str = PathParam(pattern=r'^[A-Z0-9]{10}$')):
    product = catalog().product(parent_asin)
    if product is None: raise HTTPException(404,'This product is not in the catalog.')
    return product


@app.post('/api/search')
def search(request: SearchRequest):
    started = time.perf_counter()
    try:
        selected=catalog().validate_selected_filters(request.filters.model_dump(exclude_none=True))
        query=request.query.strip()
        if not query and not selected: raise HTTPException(422,'Describe a shirt or select an optional filter.')
        required_query,preferences=search_preferences(query)
        if query:
            reason=clarification_reason(query,[b['name'] for b in catalog().filter_options()['brands']])
            if reason:
                return {'sql':'','products':[],'total_matches':0,'unsupported':[],
                    'mode':'clarification','explanation':reason,'suggestions':SUGGESTIONS,
                    'selected_filters':selected,'elapsed_seconds':round(time.perf_counter()-started,3)}
        plan = SQLGenerator().generate(required_query,metadata()) if query else {
            'sql':'SELECT parent_asin FROM shirts','explanation':'Shirts matching your selected filters.','unsupported':[]}
        if plan['unsupported']:
            return {**plan,'total_matches':0,'products':[],'mode':'unsupported',
                    'elapsed_seconds':round(time.perf_counter()-started,3)}
        plan['sql'],enforced_fields=enforce_request_sql(plan['sql'],required_query,catalog().filter_columns)
        plan['sort']=requested_sort(query)
        requested_dominant=dominant_color_intent(required_query)
        if preferences or requested_dominant:
            plan['sql']=preference_filter_sql(plan['sql'],required_query,catalog().filter_columns)
        only_color,require_plain=single_color_intent(request.query)
        if selected.get('color') and re.search(r'\b(?:plain|solid)\b',query,re.I) and not re.search(r'\bnot\s+(?:plain|solid)\b',query,re.I):
            if only_color is None:only_color=selected['color']
            require_plain=True
        results = catalog().search(plan['sql'],request.limit,request.include_uncertain,sort=plan.get('sort','recommended'),
            only_color=only_color,require_plain=require_plain,selected_filters=selected,preferences=preferences,
            dominant_color=requested_dominant)
        explanation=plan['explanation']
        if only_color:explanation+=f' Only products tagged solely {only_color} are included; mixed color tags are excluded.'
        if selected:explanation+=' Your selected filters are also required.'
        if results['requested_dominant_color']:
            explanation+=f" {results['requested_dominant_color'].capitalize()} must be the largest shade and cover at least 50% of saved shirt pixels; small accents and shadows do not qualify."
        if preferences:
            descriptions=[]
            if preferences.get('office'): descriptions.append('office-style evidence')
            if preferences.get('comfort'): descriptions.append('comfort wording and fabric/fit evidence')
            if plan['sort']=='recommended':
                explanation+=' Ranked using '+ ' and '.join(descriptions)+' from recorded listings; these are suitability signals, not guarantees.'
            else:
                explanation+=' Office/comfort signals are calculated from listings, but your explicitly requested sort takes priority; these are suitability signals, not guarantees.'
        return {**results,'explanation':explanation,'unsupported':[], 'mode':'llm_sql' if query else 'catalog_filters',
                'enforced_fields':enforced_fields,
                'elapsed_seconds':round(time.perf_counter()-started,3)}
    except LLMError as exc: raise HTTPException(503,str(exc)) from exc
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    except duckdb.Error as exc:
        # SQL execution fails explicitly; never silently rerun an unfiltered query.
        raise HTTPException(422,'Generated SQL could not run against the catalog. Try rephrasing your request.') from exc
