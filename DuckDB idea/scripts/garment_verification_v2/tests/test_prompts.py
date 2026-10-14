import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from garment_core import LABELS, decide
from classify_garments import PROMPTS
from garment_core import prepare_input, export_results
import pandas as pd
import tempfile

class PromptTests(unittest.TestCase):
    def test_one_piece_categories_are_explicit(self):
        self.assertIn('Bodysuit',LABELS)
        self.assertEqual(set(PROMPTS),set(LABELS))
        self.assertTrue(all(len(p)==3 for p in PROMPTS.values()))
        self.assertEqual(decide({'label':'Bodysuit','score':.95,'margin':.6})[0],'non_shirt_candidate')
    def test_weak_similarity_is_not_accepted(self):
        self.assertEqual(decide({'label':'Shirt','score':.3,'margin':.02})[0],'uncertain')
    def test_bodysuit_title_blocks_shirt_like_visual_prediction(self):
        products,images=prepare_input(pd.DataFrame([
            dict(parent_asin='A',image_url='https://example.com/1.jpg',title='Dance shirt leotard bodysuit'),
            dict(parent_asin='B',image_url='https://example.com/2.jpg',title='Cotton button down shirt')]))
        records={i['image_key']:{**i,'label':'Shirt','score':.95,'margin':.6} for i in images}
        with tempfile.TemporaryDirectory() as d:
            export_results(products,records,d)
            result=pd.read_parquet(Path(d)/'products_annotated.parquet')
            self.assertEqual(result.garment_status.tolist(),['non_shirt_candidate','shirt_candidate'])
            self.assertIn('Title exclusion',result.garment_reason.iloc[0])
