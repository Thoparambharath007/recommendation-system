from pathlib import Path
import sys,tempfile,unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from build_compact_catalog import compact_frames
from query_engine import Catalog
from fastapi.testclient import TestClient
from app import app

class CompactTests(unittest.TestCase):
    def test_one_product_per_photo_preserves_real_size_options_and_price_filters(self):
        rows=[]
        for asin,url,size,price,status in [('AAAAAAAAAA','same','M',900,'shirt_candidate'),
            ('BBBBBBBBBB','same','L',1600,'shirt_candidate'),('CCCCCCCCCC','other','',800,'shirt_candidate'),
            ('DDDDDDDDDD','excluded','XL',600,'uncertain')]:
            rows.append(dict(parent_asin=asin,image_url='https://example.com/'+url+'.jpg',title='Shirt '+size,
                color='blue',pattern='plain',sizes=size,brand='Brand',price_inr_est=price,
                average_rating=4.,rating_number=10,price_source='exact',garment_status=status))
        catalog,options=compact_frames(pd.DataFrame(rows))
        self.assertEqual(len(catalog),2)
        self.assertEqual(len(options),3)
        self.assertFalse(catalog.image_url.duplicated().any())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);catalog.to_parquet(p/'shirts_catalog.parquet',index=False)
            options.to_parquet(p/'shirt_size_options.parquet',index=False)
            c=Catalog(p/'shirts_catalog.parquet')
            self.assertEqual(c.overview()['products'],2)
            self.assertEqual(c.search('SELECT parent_asin FROM shirts',selected_filters={'size':'L','max_price':1200})['total_matches'],0)
            result=c.search('SELECT parent_asin FROM shirts',selected_filters={'size':'L'})
            self.assertEqual(result['products'][0]['parent_asin'],'BBBBBBBBBB')
            a,b=c.product('AAAAAAAAAA'),c.product('BBBBBBBBBB')
            self.assertEqual(a['image_url'],b['image_url'])
            self.assertEqual(b['price_inr_est'],1600)
            self.assertEqual({v['sizes'] for v in b['variants']},{'M','L'})
            self.assertEqual(c.product('CCCCCCCCCC')['sizes'],'')
            self.assertIsNone(c.product('DDDDDDDDDD'))
    def test_separate_product_page_is_served(self):
        with TestClient(app) as client:
            r=client.get('/product/AAAAAAAAAA')
            self.assertEqual(r.status_code,200)
            self.assertIn('product-content',r.text)
            self.assertEqual(client.get('/product/invalid').status_code,422)
