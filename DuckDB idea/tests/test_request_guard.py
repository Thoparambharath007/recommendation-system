import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import pandas as pd
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from request_guard import enforce_request_sql, requested_sort
from query_engine import FILTER_COLUMNS, Catalog
from app import app
from llm_sql import field_guidance

class GuardTests(unittest.TestCase):
    def test_sort_is_requested_not_guessed_from_budget(self):
        self.assertEqual(requested_sort('comfortable black shirt under 1200'),'recommended')
        self.assertEqual(requested_sort('cheapest black shirt under 1200'),'lowest_price')
        self.assertEqual(requested_sort('highest rated white shirt'),'highest_rating')
        self.assertEqual(requested_sort('most reviewed white shirt'),'most_reviewed')
        self.assertEqual(requested_sort('not the cheapest shirt'),'recommended')

    def test_missing_material_and_wrong_size_syntax_are_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'shirts_master.parquet'
            pd.DataFrame([dict(parent_asin=asin,title='Cotton office shirt',color='blue',fabric=fabric,
                sizes=size,pattern='plain',average_rating=4.,rating_number=10,price_inr_est=price,
                image_url='https://example.com/'+asin,price_source='exact') for asin,fabric,size,price in
                [('A','cotton;polyester','M',900),('B','polyester','M',900),('C','cotton','XL',900),
                 ('D','cotton','M',1900)]]).to_parquet(p,index=False)
            c=Catalog(p)
            sql="SELECT parent_asin FROM shirts WHERE color='blue' AND sizes LIKE '%|m|%'"
            fixed,_=enforce_request_sql(sql,'blue cotton shirt size M under 1500 rupees',c.filter_columns)
            result=c.search(fixed)
            self.assertEqual([x['parent_asin'] for x in result['products']],['A'])
            wrong="SELECT parent_asin FROM shirts WHERE contains('|' || fabric || '|','|cotton|')"
            fixed,_=enforce_request_sql(wrong,'blue cotton shirt size M under 1500',c.filter_columns)
            self.assertEqual(c.search(fixed)['total_matches'],1)

    def test_preserves_alternatives_negation_and_uncontrolled_constraints(self):
        sql="SELECT parent_asin FROM shirts WHERE (color='red' OR average_rating>4)"
        with self.assertRaises(ValueError):enforce_request_sql(sql,'blue shirt',FILTER_COLUMNS)
        fixed,_=enforce_request_sql('SELECT parent_asin FROM shirts WHERE average_rating>4',
            'black or white cotton or linen shirt',FILTER_COLUMNS)
        self.assertIn('OR',fixed)
        self.assertIn('average_rating > 4',fixed)
        fixed,_=enforce_request_sql("SELECT parent_asin FROM shirts WHERE color!='black'",
            'shirt not black',FILTER_COLUMNS)
        self.assertIn('NOT',fixed)
        self.assertIn('black',fixed)
        with self.assertRaises(ValueError):enforce_request_sql("SELECT parent_asin FROM read_csv('private')",'black shirt',FILTER_COLUMNS)

    def test_field_examples_use_the_recorded_delimiters(self):
        guide=field_guidance({'columns':[{'name':'fabric','type':'VARCHAR'},{'name':'sizes','type':'VARCHAR'}]})
        self.assertEqual(set(guide),{'fabric','sizes'})
        self.assertIn(';cotton;',guide['fabric']['sql_example'])
        self.assertIn('|m|',guide['sizes']['sql_example'])

    def test_cross_field_alternatives_do_not_expand_into_wrong_combinations(self):
        with self.assertRaises(ValueError):
            enforce_request_sql('SELECT parent_asin FROM shirts','black cotton or white linen shirts',FILTER_COLUMNS)

    def test_currency_and_explicit_upper_lower_bounds(self):
        sql,_=enforce_request_sql('SELECT parent_asin FROM shirts','white shirt above 10 dollars under 25 dollars',FILTER_COLUMNS)
        self.assertIn('price_usd > 10',sql)
        self.assertIn('price_usd < 25',sql)
        self.assertNotIn('price_inr_est',sql)

    def test_model_price_sort_does_not_bypass_office_comfort(self):
        plan=dict(sql="SELECT parent_asin FROM shirts WHERE color='black' AND price_inr_est<1200",
            explanation='Black shirts',sort='lowest_price',unsupported=[])
        with patch('app.SQLGenerator.generate',return_value=plan),TestClient(app) as client:
            r=client.post('/api/search',json={'query':'comfortable black shirt for office under 1200'})
            self.assertEqual(r.status_code,200,r.text)
            j=r.json();self.assertEqual(j['sort'],'recommended')
            scores=[p['preference_score'] for p in j['products']]
            self.assertEqual(scores,sorted(scores,reverse=True))
            self.assertEqual(client.post('/api/search',json={'query':'cheapest black shirt under 1200'}).json()['sort'],'lowest_price')
