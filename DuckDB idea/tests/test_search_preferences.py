from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import app
from query_engine import Catalog
from request_policy import search_preferences


class PreferenceTests(unittest.TestCase):
    def test_typo_query_keeps_constraints_and_extracts_preferences(self):
        query, preferences = search_preferences('comfortable plain white shrt full sleeve for officie work under 1200')
        self.assertEqual(preferences, {'office': True, 'comfort': True})
        self.assertIn('plain white shirt full sleeve', query)
        self.assertIn('under 1200', query)
        self.assertNotIn('office', query)
        self.assertNotIn('comfortable', query)
        self.assertEqual(search_preferences('black shirt with white buttons')[1], {})
        self.assertEqual(search_preferences('black shirt not for office')[1], {})

    def test_preferences_rank_all_matches_without_changing_required_filters(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'JUNK created by qoder') as directory:
            path = Path(directory)/'shirts_master.parquet'
            rows = []
            for asin, title, color, price, fabric in [
                ('A', 'Black cotton dress shirt button down regular fit', 'black', 900, 'cotton'),
                ('B', 'Black Hawaiian beach shirt', 'black', 900, 'polyester'),
                ('C', 'White cotton office dress shirt', 'white', 900, 'cotton'),
                ('D', 'Black office shirt', 'black', 1900, 'cotton')]:
                rows.append(dict(parent_asin=asin,title=title,color=color,pattern='plain',fabric=fabric,
                    fit_type='regular fit',average_rating=4.0,rating_number=20,
                    price_source='estimated',price_inr_est=price))
            pd.DataFrame(rows).to_parquet(path,index=False)
            catalog = Catalog(path)
            sql = "SELECT parent_asin FROM shirts WHERE color='black' AND price_inr_est < 1200"
            result = catalog.search(sql, preferences={'office':True,'comfort':True})
            self.assertEqual(result['total_matches'], 2)
            self.assertEqual([p['parent_asin'] for p in result['products']], ['A','B'])
            self.assertGreater(result['products'][0]['preference_score'],result['products'][1]['preference_score'])
            self.assertIn('office',result['ranking'].lower())
            # An explicit price sort remains the primary ordering.
            self.assertEqual(catalog.search(sql,sort='lowest_price',preferences={'office':True})['sort'],'lowest_price')

    def test_api_sends_only_required_constraints_to_llm(self):
        plan = dict(sql="SELECT parent_asin FROM shirts WHERE color='black' AND price_inr_est < 1200",
                    explanation='Black shirts under 1200 INR.',unsupported=[])
        with patch('app.SQLGenerator.generate',return_value=plan) as generate, TestClient(app) as client:
            response = client.post('/api/search',json={'query':'comfortable black shirt for office under 1200'})
            self.assertEqual(response.status_code,200,response.text)
            data=response.json()
            self.assertGreater(data['total_matches'],0)
            self.assertEqual(data['preferences'],{'office':True,'comfort':True})
            self.assertTrue(all('black' in p['color'].split('|') and p['price_inr_est']<1200 for p in data['products']))
            self.assertNotIn('office',generate.call_args.args[0])
            self.assertNotIn('comfortable',generate.call_args.args[0])
            self.assertIn('not guarantees',data['explanation'])

    def test_unsupported_required_attribute_is_still_rejected(self):
        plan=dict(sql='',explanation='Button color unavailable',unsupported=['button color'])
        with patch('app.SQLGenerator.generate',return_value=plan), TestClient(app) as client:
            data=client.post('/api/search',json={'query':'comfortable black shirt with white buttons for office'}).json()
            self.assertEqual(data['mode'],'unsupported')
            self.assertEqual(data['products'],[])

    def test_preference_queries_do_not_inherit_unrequested_plain_filter(self):
        plan=dict(sql="SELECT parent_asin FROM shirts WHERE color='black' AND pattern='plain'",
                  explanation='Black shirts',unsupported=[])
        with patch('app.SQLGenerator.generate',return_value=plan),TestClient(app) as client:
            data=client.post('/api/search',json={'query':'black shirt for office'}).json()
            expected=Catalog().search("SELECT parent_asin FROM shirts WHERE color='black'",dominant_color='black')['total_matches']
            self.assertEqual(data['total_matches'],expected)
            plain=client.post('/api/search',json={'query':'plain black shirt for office'}).json()
            self.assertTrue(all(p['pattern']=='plain' and p['color']=='black' for p in plain['products']))


if __name__ == '__main__': unittest.main()
