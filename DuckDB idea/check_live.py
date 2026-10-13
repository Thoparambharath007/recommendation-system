"""Small real-provider smoke check; never print credentials or provider raw errors."""
import json
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env',override=False)
from llm_sql import SQLGenerator, LLMError
from query_engine import Catalog

if __name__=='__main__':
    c=Catalog(); generator=SQLGenerator()
    query='Find maroon plain full sleeve shirts'
    print('Checking live LLM SQL generation...',flush=True)
    try:
        plan=generator.generate(query,c.overview())
        if plan['unsupported']: raise ValueError('Supported smoke query was rejected')
        result=c.search(plan['sql'],3)
        report={'query':query,'model':generator.model,'sql':result['sql'],
                'matches':result['total_matches'],'ids':[r['parent_asin'] for r in result['products']],
                'live_llm_passed':True}
        print(json.dumps(report,indent=2))
    except (LLMError,ValueError) as exc:
        print(json.dumps({'live_llm_passed':False,'error':str(exc)}))
        raise SystemExit(1)
