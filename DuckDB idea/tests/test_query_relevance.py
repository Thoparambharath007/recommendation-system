from pathlib import Path
import sys,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app import app
from fastapi.testclient import TestClient

class RelevanceGateTests(unittest.TestCase):
    def test_unknown_text_never_calls_model_or_returns_products(self):
        with patch('app.SQLGenerator.generate') as generate,TestClient(app) as client:
            for query in ['Kollah','asdjklqwerty','!!!','hello','what is the weather','blue shoes']:
                r=client.post('/api/search',json={'query':query})
                self.assertEqual(r.status_code,200,r.text)
                d=r.json();self.assertEqual(d['mode'],'clarification',query)
                self.assertEqual(d['products'],[])
                self.assertEqual(d['sql'],'')
                self.assertTrue(d['suggestions'])
            generate.assert_not_called()
    def test_valid_short_queries_brands_and_filters_still_work(self):
        plan=dict(sql='SELECT parent_asin FROM shirts',explanation='Shirts',unsupported=[])
        with patch('app.SQLGenerator.generate',return_value=plan),TestClient(app) as client:
            for query in ['white','comfortable for office','full sleeve','shirts','size M','cotton','XL','highest rating']:
                r=client.post('/api/search',json={'query':query})
                self.assertEqual(r.status_code,200,r.text)
                self.assertNotEqual(r.json()['mode'],'clarification',query)
            self.assertEqual(client.post('/api/search',json={'query':'','filters':{'color':'black'}}).status_code,200)
            brands=client.get('/api/filters').json()['brands']
            self.assertNotEqual(client.post('/api/search',json={'query':brands[0]['name']}).json()['mode'],'clarification')
    def test_unrecognized_text_is_not_ignored_when_filters_are_selected(self):
        with patch('app.SQLGenerator.generate') as generate,TestClient(app) as client:
            d=client.post('/api/search',json={'query':'Kollah','filters':{'color':'black'}}).json()
            self.assertEqual(d['mode'],'clarification');self.assertEqual(d['products'],[])
            generate.assert_not_called()
