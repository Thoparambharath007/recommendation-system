import sys
from pathlib import Path
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from product_enrichment import extract_sizes, enrich_frames
from query_engine import Catalog, validate_sql
from fastapi.testclient import TestClient
from app import app


class SizeTests(unittest.TestCase):
    def test_structured_size_and_explicit_size_chart(self):
        self.assertEqual(extract_sizes('Shirt', {'Size':'X-Large'}, [], [])['sizes'], 'XL')
        self.assertEqual(extract_sizes('Shirt', {}, ['Sizes: S, M, L, XL, 2XL'], [])['sizes'], 'S|M|L|XL|XXL')
        self.assertEqual(extract_sizes('Shirt', {}, ['S=(US 37.5), M=(US 40), L=(US 43), XL=(US 46), 2XL=(US 48.5).'], [])['sizes'], 'S|M|L|XL|XXL')
        self.assertEqual(extract_sizes('Cotton Shirt, Large', {}, [], [])['sizes'], 'L')
        self.assertEqual(extract_sizes('Shirt', {}, ['Available sizes: Small (Neck Size 14), Medium (Neck Size 15), Large (Neck Size 16), XL (Neck Size 17), 2XL (Neck Size 18)'], [])['sizes'], 'S|M|L|XL|XXL')

    def test_unknown_measurements_model_size_and_other_products_are_not_sizes(self):
        for details, features in [({'Wheel Size':'Large'},[]), ({'Package Dimensions':'M L XL'},[]),
                                  ({},['The model wears a size Medium.']),
                                  ({},['Large pockets and medium weight fabric']),
                                  ({},['Size Small, Please Choose 1 Or 2 Size Up']),
                                  ({},['Machine wash; choose one size up'])]:
            with self.subTest(features=features):
                self.assertEqual(extract_sizes('Classic shirt', details, features, [])['sizes'], '')
        numeric=extract_sizes('Shirt', {'Size':'16.5 Neck 34-35 Sleeve'}, [], [])
        self.assertEqual(numeric['sizes'], '')
        self.assertIn('16.5', numeric['size_evidence'])
        self.assertEqual(extract_sizes("Rip Curl Men's Barney L/S Shirt", {}, [], [])['sizes'], '')
        self.assertEqual(extract_sizes("The North Face Men's S/S Shirt, XL", {}, [], [])['sizes'], 'XL')

    def test_no_all_sizes_default_and_no_range_expansion(self):
        self.assertEqual(extract_sizes('Shirt', {}, [], [])['size_source'], 'unavailable')
        self.assertEqual(extract_sizes('Shirt', {}, ['Sizes S-XXL'], [])['sizes'], 'S|XXL')

    def test_enrichment_keeps_master_and_aligns_product_details_by_id(self):
        master=pd.DataFrame([{'parent_asin':'B','title':'B shirt','store':'Store B','color':'pink','sleeve':'full'},
                             {'parent_asin':'A','title':'A shirt','store':'Store A','color':'black','sleeve':'half'}])
        audit=pd.DataFrame([{'parent_asin':'A','source_title':'A shirt','source_details':'{"Size":"Medium","Brand":"Brand A"}',
                             'source_features':'["Cotton"]','source_description':'["A description"]',
                             'source_image_urls':'["https://example.com/a.jpg"]'},
                            {'parent_asin':'B','source_title':'B shirt','source_details':'{}',
                             'source_features':'[]','source_description':'[]','source_image_urls':'[]'}])
        brands=pd.DataFrame({'parent_asin':['B'], 'brand':['Brand B']})
        enriched, details=enrich_frames(master,audit,brands)
        pd.testing.assert_frame_equal(enriched[master.columns],master)
        self.assertEqual(enriched.sizes.tolist(),['','M'])
        self.assertEqual(enriched.brand.tolist(),['Brand B','Brand A'])
        self.assertEqual(details.parent_asin.tolist(),['B','A'])
        with self.assertRaises(ValueError): enrich_frames(master,audit.iloc[:1],brands)


class ProductDetailsTests(unittest.TestCase):
    def test_size_filters_use_exact_tokens_even_for_llm_like_and_aliases(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'JUNK created by qoder') as directory:
            path=Path(directory)/'shirts_master.parquet'
            pd.DataFrame([{'parent_asin':size,'sizes':size,'title':size,'color':'pink','pattern':'plain',
                           'average_rating':4.0,'rating_number':12,'price_source':'exact','price_inr_est':1000}
                          for size in ['M','XL','XXL']]).to_parquet(path,index=False)
            catalog=Catalog(path)
            for predicate in ["sizes LIKE '%M%'", "lower(sizes) = 'medium'", "sizes IN ('M')"]:
                result=catalog.search('SELECT parent_asin FROM shirts WHERE '+predicate)
                self.assertEqual([p['parent_asin'] for p in result['products']],['M'])
            with self.assertRaises(ValueError): validate_sql("SELECT parent_asin FROM shirts WHERE sizes='invented'",catalog.filter_columns)

    def test_http_detail_routes_and_assets(self):
        with TestClient(app) as client:
            active_id=Catalog().search('SELECT parent_asin FROM shirts',limit=1)['products'][0]['parent_asin']
            self.assertEqual(client.get('/api/products/'+active_id).status_code,200)
            self.assertEqual(client.get('/api/products/ZZZZZZZZZZ').status_code,404)
            self.assertEqual(client.get('/api/products/invalid').status_code,422)
            self.assertEqual(client.get('/assets/product_details.js').status_code,200)
            self.assertEqual(client.get('/assets/../.env').status_code,404)

    def test_product_details_lookup_is_parameterized_and_combines_correct_id(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'JUNK created by qoder') as directory:
            path=Path(directory)
            pd.DataFrame([{'parent_asin':'A','title':'A','average_rating':4.0,'color':'pink','sizes':'M'}]).to_parquet(path/'shirts_master.parquet',index=False)
            pd.DataFrame([{'parent_asin':'A','features_json':'["Cotton"]','description_json':'["Description A"]',
                           'details_json':'{"Size":"Medium"}','image_urls_json':'[]'}]).to_parquet(path/'product_details.parquet',index=False)
            c=Catalog(path/'shirts_master.parquet')
            self.assertEqual(c.product('A')['features'],['Cotton'])
            self.assertEqual(c.product('A')['sizes'],'M')
            self.assertIsNone(c.product("A' OR 1=1 --"))


if __name__=='__main__': unittest.main()
