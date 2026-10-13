"""Local natural-language apparel search over the merged Parquet catalog."""
from functools import lru_cache
from pathlib import Path
import time

import duckdb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=False)
from query_engine import Catalog
from llm_sql import SQLGenerator, LLMError
from request_policy import single_color_intent

app = FastAPI(title='Shirt catalog: LLM + DuckDB', docs_url='/docs')


@lru_cache(maxsize=1)
def catalog(): return Catalog()


@lru_cache(maxsize=1)
def metadata(): return catalog().overview()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=12, ge=1, le=50)
    include_uncertain: bool = True


@app.get('/')
def index(): return FileResponse(ROOT / 'web/index.html')


@app.get('/api/health')
def health():
    generator = SQLGenerator()
    return {'ready':True, **metadata(), 'llm_configured':bool(generator.api_key), 'model':generator.model}


@app.get('/api/browse')
def browse(limit: int = Query(default=12,ge=1,le=50)):
    return {**catalog().search('SELECT parent_asin FROM shirts',limit=limit),
            'explanation':'Browse the catalog, ranked by price provenance and rating evidence.',
            'unsupported':[], 'mode':'catalog_browse'}


@app.post('/api/search')
def search(request: SearchRequest):
    started = time.perf_counter()
    if not request.query.strip(): raise HTTPException(422,'Enter a shirt request.')
    try:
        plan = SQLGenerator().generate(request.query.strip(),metadata())
        if plan['unsupported']:
            return {**plan,'total_matches':0,'products':[],'mode':'unsupported',
                    'elapsed_seconds':round(time.perf_counter()-started,3)}
        only_color,require_plain=single_color_intent(request.query)
        results = catalog().search(plan['sql'],request.limit,request.include_uncertain,sort=plan.get('sort','recommended'),
            only_color=only_color,require_plain=require_plain)
        explanation=plan['explanation']
        if only_color:explanation+=f' Only products tagged solely {only_color} are included; mixed color tags are excluded.'
        return {**results,'explanation':explanation,'unsupported':[], 'mode':'llm_sql',
                'elapsed_seconds':round(time.perf_counter()-started,3)}
    except LLMError as exc: raise HTTPException(503,str(exc)) from exc
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    except duckdb.Error as exc:
        # SQL execution fails explicitly; never silently rerun an unfiltered query.
        raise HTTPException(422,'Generated SQL could not run against the catalog. Try rephrasing your request.') from exc
