"""Capture real configured-model SQL and application ranking without exposing keys."""
from pathlib import Path
import sys
import json
import argparse
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import app, metadata
from llm_sql import SQLGenerator, LLMError
from fastapi.testclient import TestClient

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='docs/verification/live_preference_sql_audit.json')
    args=parser.parse_args()
    generator=SQLGenerator()
    schema=metadata()
    report={'model':generator.model,'schema_sent':schema,'tests':[]}
    queries=[
        'a comfortable white shirt for office use',
        'comfortable black shirt for office under 1200 rupees',
        'a white shirt for office work full sleeve',
        'comfortable blue cotton shirt size M under 1500 rupees',
        'plain pink shirt full sleeve under 1200 rupees',
    ]
    original=SQLGenerator.generate
    destination=ROOT/args.output
    destination.parent.mkdir(parents=True,exist_ok=True)
    def save():destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    # An additional direct-model call establishes what happens without stripping.
    try:
        report['direct_full_query']={'query':queries[0],'plan':generator.generate(queries[0],schema)}
    except LLMError as e:report['direct_full_query']={'query':queries[0],'error':str(e)}
    save()
    print('DIRECT FULL QUERY:',json.dumps(report['direct_full_query']),flush=True)
    for query in queries:
        capture={}
        def observed(self,model_query,catalog_schema):
            capture['query_sent_to_model']=model_query
            plan=original(self,model_query,catalog_schema)
            capture['raw_generated_plan']=plan.copy()
            return plan
        with patch.object(SQLGenerator,'generate',observed),TestClient(app) as client:
            response=client.post('/api/search',json={'query':query,'limit':3})
        data=response.json()
        item={'user_query':query,**capture,'http_status':response.status_code}
        if response.status_code==200:
            item.update(executed_sql=data.get('sql'),preferences=data.get('preferences'),
                total_matches=data.get('total_matches'),explanation=data.get('explanation'),
                ranking_note=data.get('ranking_note'),unsupported=data.get('unsupported'),
                enforced_fields=data.get('enforced_fields'),sort=data.get('sort'),ranking=data.get('ranking'),
                top_products=[{k:p.get(k) for k in ['parent_asin','title','color','color_main','sleeve','fabric','fit_type','style','price_inr_est','preference_score']} for p in data.get('products',[])])
        else:item['error']=data
        report['tests'].append(item);save()
        print(json.dumps(item,indent=2),flush=True)
    print('Saved:',destination,flush=True)

if __name__=='__main__':main()
