from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import app
from query_engine import Catalog


class SelectedFilterTests(unittest.TestCase):
    def test_catalog_selections_intersect_query_and_unselected_fields_are_unrestricted(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'JUNK created by qoder') as directory:
            path=Path(directory)/'shirts_master.parquet'
            pd.DataFrame([dict(parent_asin=asin,title=asin,brand=brand,color=color,pattern='plain',
                sizes=size,sleeve=sleeve,fabric='cotton',fit_type='regular fit',pocket='single',
                average_rating=4.0,rating_number=20,price_source='exact',price_inr_est=price)
                for asin,brand,color,size,sleeve,price in [
                    ('A',"O'Neill",'black','M','full',900),('B','Other','black','XL','half',900),
                    ('C',"O'Neill",'blue','M','full',900),('D',"O'Neill",'black','M','full',1700),
                    ('E','Other','navy|black','M','full',900)]]).to_parquet(path,index=False)
            catalog=Catalog(path)
            sql='SELECT parent_asin FROM shirts WHERE price_inr_est < 1200'
            self.assertEqual(catalog.search(sql,selected_filters={})['total_matches'],4)
            result=catalog.search(sql,selected_filters={'brands':["O'Neill"],'color':'black','size':'M','sleeve':'full'})
            self.assertEqual([p['parent_asin'] for p in result['products']],['A'])
            multi=catalog.search(sql,selected_filters={'brands':["O'Neill",'Other'],'color':'black','pattern':'plain'})
            self.assertEqual({p['parent_asin'] for p in multi['products']},{'A','B'})
            self.assertEqual(catalog.search(sql+" AND color='blue'",selected_filters={'color':'black'})['total_matches'],0)
            self.assertEqual(catalog.search('SELECT parent_asin FROM shirts',selected_filters={'max_price':950})['total_matches'],4)
            with self.assertRaises(ValueError): catalog.search(sql,selected_filters={'brands':["x' OR 1=1 --"]})

    def test_api_returns_complete_brand_list_and_choices(self):
        with TestClient(app) as client:
            response=client.get('/api/filters')
            self.assertEqual(response.status_code,200)
            options=response.json()
            self.assertTrue(any(b['name']=='JMIERR' for b in options['brands']))
            self.assertGreater(len(options['brands']),100)
            self.assertEqual(len({b['name'].casefold() for b in options['brands']}),len(options['brands']))
            self.assertIn('M',options['sizes'])

    def test_selected_brand_is_enforced_even_when_llm_does_not_mention_it(self):
        plan={'sql':'SELECT parent_asin FROM shirts WHERE price_inr_est < 2000','explanation':'Budget shirts','unsupported':[]}
        with patch('app.SQLGenerator.generate',return_value=plan),TestClient(app) as client:
            response=client.post('/api/search',json={'query':'shirts under 2000','filters':{'brands':['JMIERR']},'limit':50})
            self.assertEqual(response.status_code,200,response.text)
            self.assertGreater(response.json()['total_matches'],0)
            self.assertTrue(all(p['brand'].lower()=='jmierr' and p['price_inr_est']<2000 for p in response.json()['products']))

    def test_filter_only_search_does_not_need_llm_and_invalid_choices_are_rejected(self):
        with patch('app.SQLGenerator.generate') as generate,TestClient(app) as client:
            response=client.post('/api/search',json={'filters':{'brands':['JMIERR'],'size':'M'}})
            self.assertEqual(response.status_code,200,response.text)
            self.assertGreater(response.json()['total_matches'],0)
            generate.assert_not_called()
            for filters in [{'brands':['Not a real brand']},{'color':'made up'},{'sleeve':'bogus'},{'max_price':-1},{'extra_field':'bad'}]:
                self.assertEqual(client.post('/api/search',json={'query':'shirts','filters':filters}).status_code,422)


if __name__=='__main__': unittest.main()
