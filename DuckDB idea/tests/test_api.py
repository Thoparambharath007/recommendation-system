import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from app import app


class APITests(unittest.TestCase):
    def test_pink_full_sleeve_budget_regression(self):
        plan=dict(sql="SELECT parent_asin FROM shirts WHERE color='pink' AND pattern='plain' AND sleeve='full' AND price_inr_est < 1200",
                  explanation='Plain pink long sleeve shirts under 1200 INR',unsupported=[])
        with patch('app.SQLGenerator.generate',return_value=plan),TestClient(app) as client:
            response=client.post('/api/search',json={'query':'plain pink shirt full sleeve under 1200','limit':50})
            self.assertEqual(response.status_code,200,response.text)
            data=response.json()
            self.assertTrue({'B005CHVJJ2','B005CHVJQK','B005CHVK50'} <= {p['parent_asin'] for p in data['products']})
            self.assertTrue(all(p['sleeve']=='full' and p['color']=='pink' and p['pattern']=='plain' and p['price_inr_est']<1200 for p in data['products']))

    def test_plain_black_rejects_mixed_colors_even_when_llm_uses_contains(self):
        plan=dict(sql="SELECT parent_asin FROM shirts WHERE contains('|'||color||'|','|black|') AND pattern='plain' AND price_inr_est < 1500",
                  explanation='Plain black shirts under 1500 INR',unsupported=[])
        with patch('app.SQLGenerator.generate',return_value=plan),TestClient(app) as client:
            data=client.post('/api/search',json={'query':'plain black shirt uder 1500','limit':50}).json()
            self.assertGreater(data['total_matches'],0)
            self.assertTrue(all(p['color']=='black' and p['pattern']=='plain' and p['price_inr_est']<1500 for p in data['products']))

    def test_real_catalog_health_and_browse(self):
        with TestClient(app) as client:
            health = client.get('/api/health').json()
            self.assertEqual(health['products'],50224)
            browse = client.get('/api/browse?limit=3')
            self.assertEqual(browse.status_code,200)
            self.assertEqual(len(browse.json()['products']),3)
            self.assertIn('image_url',browse.json()['products'][0])
            self.assertEqual(client.get('/').status_code,200)

    def test_llm_sql_is_validated_before_real_duckdb_search(self):
        plan = dict(sql="SELECT parent_asin FROM shirts WHERE contains('|' || color || '|','|maroon|')",
                    explanation='Maroon shirts',unsupported=[])
        with patch('app.SQLGenerator.generate',return_value=plan), TestClient(app) as client:
            response = client.post('/api/search',json={'query':'maroon shirts','limit':5})
            self.assertEqual(response.status_code,200,response.text)
            data = response.json()
            self.assertGreater(data['total_matches'],0)
            self.assertTrue(all('maroon' in x['color'].split('|') for x in data['products']))
        with patch('app.SQLGenerator.generate',return_value={**plan,'sql':"SELECT parent_asin FROM read_csv('secret')"}), TestClient(app) as client:
            self.assertEqual(client.post('/api/search',json={'query':'shirts'}).status_code,422)

    def test_unsupported_attribute_does_not_silently_drop_filter(self):
        plan = dict(sql='',explanation='Button color is unavailable',unsupported=['button color'])
        with patch('app.SQLGenerator.generate',return_value=plan), TestClient(app) as client:
            data = client.post('/api/search',json={'query':'black shirt with white buttons'}).json()
            self.assertEqual(data['products'],[])
            self.assertEqual(data['unsupported'],['button color'])

    def test_invalid_input(self):
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/search',json={'query':''}).status_code,422)
            self.assertEqual(client.post('/api/search',json={'query':'shirts','limit':10000}).status_code,422)


if __name__=='__main__':unittest.main()
