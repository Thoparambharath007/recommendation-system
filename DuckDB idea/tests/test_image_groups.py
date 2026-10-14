from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from query_engine import Catalog


class ImageGroupTests(unittest.TestCase):
    def test_grouping_precedes_limit_and_uses_a_matching_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'shirts_master.parquet'
            rows=[]
            for asin,image,size,price,rating in [('A','same','M',900,5.),('B','same','L',1000,4.),('C','other','L',800,3.)]:
                rows.append(dict(parent_asin=asin,image_url='https://example.com/'+image+'.jpg',sizes=size,
                    brand='Brand',title='Shirt '+size,color='blue',pattern='plain',pocket='double',
                    average_rating=rating,rating_number=30,price_source='exact',price_inr_est=price))
            pd.DataFrame(rows).to_parquet(path,index=False)
            c=Catalog(path)
            result=c.search('SELECT parent_asin FROM shirts',limit=2)
            self.assertEqual(result['total_matches'],2)
            self.assertEqual(len({p['image_url'] for p in result['products']}),2)
            selected=c.search('SELECT parent_asin FROM shirts',selected_filters={'size':'L'},limit=2)
            self.assertEqual({p['parent_asin'] for p in selected['products']},{'B','C'})
            details=c.product('B')
            self.assertEqual({v['parent_asin'] for v in details['variants']},{'A','B'})
            self.assertEqual(details['sizes'],'L')
            self.assertEqual(details['price_inr_est'],1000)

    def test_missing_images_do_not_collapse_unrelated_products(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'shirts_master.parquet'
            pd.DataFrame([dict(parent_asin=asin,image_url=None,title='Shirt',color='black',pattern='plain',
                average_rating=4.,rating_number=2,price_source='exact',price_inr_est=800) for asin in ['A','B']]).to_parquet(path,index=False)
            self.assertEqual(Catalog(path).search('SELECT parent_asin FROM shirts')['total_matches'],2)


if __name__=='__main__':unittest.main()
