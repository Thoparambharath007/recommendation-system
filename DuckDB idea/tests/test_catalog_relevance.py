import json
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
from request_policy import dominant_color_intent
from catalog_quality import non_shirt_reason, palette_dominance


class RelevanceTests(unittest.TestCase):
    def test_bodysuits_are_not_regular_shirts(self):
        self.assertTrue(non_shirt_reason('iEFIEL Mens Long Sleeve Latin Modern Dance Shirt with Bowtie One-piece Leotard Bodysuit Romper Shirt'))
        self.assertTrue(non_shirt_reason('Men dance bodysuit shirt'))
        self.assertFalse(non_shirt_reason('Cotton shirt to wear with a bodysuit'))
        self.assertFalse(non_shirt_reason('Cotton button down shirt'))
    def test_bad_or_ambiguous_palette_does_not_create_dominance(self):
        for value in ['bad',None,'[]','[{"name":"black","share":1.5}]',
                      '[{"name":"black","share":0.5},{"name":"white","share":0.5}]',
                      '[{"name":"black","share":0.8},{"name":"white","share":0.8}]']:
            self.assertEqual(palette_dominance(value),(None,None))
        self.assertEqual(palette_dominance('[{"name":"gold","share":0.7},{"name":"yellow","share":0.3}]'),('gold',.7))
    def test_single_color_intent_preserves_multicolor_and_negative_requests(self):
        self.assertEqual(dominant_color_intent('black shirt under 1200 rupees'),'black')
        self.assertEqual(dominant_color_intent('comfortable sky blue shirt for office'),'light blue')
        self.assertIsNone(dominant_color_intent('black and white striped shirts'))
        self.assertIsNone(dominant_color_intent('black or navy shirt'))
        self.assertIsNone(dominant_color_intent('not black shirts'))

    def test_primary_outerwear_is_excluded_without_matching_incidental_words(self):
        self.assertTrue(non_shirt_reason('Jackets for Men,Men Casual Blazer Jackets Slim Fit Suits Jacket One Button Business Sports Coat Solid Long Sleeve Shirts'))
        self.assertTrue(non_shirt_reason('Mens Denim Jacket Lightweight Button Down'))
        self.assertFalse(non_shirt_reason("G.H. Bass Men's Plaid Shirt, Rich Navy Blazer, X-Large"))
        self.assertFalse(non_shirt_reason('Cotton dress shirt to wear under a jacket'))
        self.assertFalse(non_shirt_reason('Jacket style button down shirt'))
        self.assertFalse(non_shirt_reason('Soft flannel shirt jacket'))
        self.assertFalse(non_shirt_reason("Pepe Jeans Men's Cotton Button Down Shirt"))
        self.assertFalse(non_shirt_reason("Arrow Men's Blazer Long Sleeve Button Down Plaid Shirt"))
        self.assertFalse(non_shirt_reason("Calvin Klein Men's Short Sleeve Coat Front Shirt"))
        self.assertFalse(non_shirt_reason("Men's Plaid Hoodie Flannel Shirt"))

    def test_dominance_filters_before_count_and_limit_and_keeps_shades_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'shirts_master.parquet'
            rows=[]
            for asin,title,color,shares in [
                ('A','Cotton shirt','black|white',[('black',.8),('white',.2)]),
                ('B','Floral shirt','white|black',[('white',.8),('black',.2)]),
                ('C','Satin shirt','navy|black',[('navy',.8),('black',.2)]),
                ('D','Blazer Jacket Long Sleeve Shirts','black',[('black',1.)]),
                ('E','Black shirt','black|charcoal',[('black',.4),('charcoal',.6)]),
                ('F','Black shirt','black',[('black',1.)])]:
                rows.append(dict(parent_asin=asin,title=title,color=color,color_main=shares[0][0],pattern='plain',
                    color_candidates=json.dumps([dict(name=n,share=s) for n,s in shares]),
                    average_rating=4.,rating_number=20,price_source='exact',price_inr_est=900))
            pd.DataFrame(rows).to_parquet(path,index=False)
            c=Catalog(path)
            sql="SELECT parent_asin FROM shirts WHERE color='black' AND price_inr_est < 1200"
            result=c.search(sql,dominant_color='black')
            self.assertEqual(result['total_matches'],2)
            self.assertEqual({p['parent_asin'] for p in result['products']},{'A','F'})
            self.assertTrue(all(p['requested_color_share']>=.5 for p in result['products']))
            self.assertEqual(c.search(sql,limit=1,dominant_color='black')['products'][0]['parent_asin'],'F')
            # Explicit black + white stays multi-color, not a dominant-color filter.
            self.assertEqual(c.search(sql+" AND color='white'")['total_matches'],2)
            self.assertNotIn('D',{p['parent_asin'] for p in c.search('SELECT parent_asin FROM shirts')['products']})

    def test_real_query_rejects_reported_examples_without_trusting_llm(self):
        plan=dict(sql="SELECT parent_asin FROM shirts WHERE color='black' AND price_inr_est < 1200",
                  explanation='Black shirts under 1200 INR',unsupported=[])
        with patch('app.SQLGenerator.generate',return_value=plan),TestClient(app) as client:
            response=client.post('/api/search',json={'query':'black shirt under 1200 rupees','limit':50})
            self.assertEqual(response.status_code,200,response.text)
            result=response.json()
            self.assertGreater(result['total_matches'],0)
            self.assertTrue(all(p['requested_color_share']>=.5 for p in result['products']))
            self.assertTrue(all(p['color_main']=='black' for p in result['products']))
            self.assertFalse({'B0B9BQ867Q','B09HMXGBHQ','B0BJFDV3CG'} & {p['parent_asin'] for p in result['products']})


if __name__=='__main__': unittest.main()
