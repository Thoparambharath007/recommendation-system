import json
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from garment_core import prepare_input, decide, Checkpoint, export_results, evaluate_labels


class CoreTests(unittest.TestCase):
    def test_unique_urls_retain_every_product_and_reject_bad_ids(self):
        frame=pd.DataFrame([dict(parent_asin='A',image_url='https://m.media-amazon.com/a.jpg'),
                            dict(parent_asin='B',image_url='https://m.media-amazon.com/a.jpg'),
                            dict(parent_asin='C',image_url=None)])
        products,images=prepare_input(frame)
        self.assertEqual(len(products),3)
        self.assertEqual(len(images),1)
        for bad in [pd.concat([frame,frame.iloc[:1]]),frame.assign(parent_asin=['A',None,'C'])]:
            with self.assertRaises(ValueError):prepare_input(bad)

    def test_decisions_do_not_accept_tshirts_or_low_margin_predictions(self):
        self.assertEqual(decide({'label':'Shirt','score':.95,'margin':.4})[0],'shirt_candidate')
        self.assertEqual(decide({'label':'T-shirt','score':.95,'margin':.4})[0],'non_shirt_candidate')
        self.assertEqual(decide({'label':'Polo','score':.95,'margin':.4})[0],'non_shirt_candidate')
        self.assertEqual(decide({'label':'Shirt','score':.5,'margin':.4})[0],'uncertain')
        self.assertEqual(decide({'label':'Shirt','score':.9,'margin':.04})[0],'uncertain')
        self.assertEqual(decide({'error':'timeout'})[0],'download_failed')

    def test_checkpoint_resume_is_atomic_and_model_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with Checkpoint(directory,{'model':'one'}) as state:
                state.put_many([{'image_key':'K','image_url':'URL','label':'Shirt','score':.95,'margin':.5}])
            with Checkpoint(directory,{'model':'one'}) as state:
                self.assertIn('K',state.keys())
                self.assertEqual(state.results()['K']['label'],'Shirt')
            with self.assertRaises(ValueError):Checkpoint(directory,{'model':'other'})

    def test_export_maps_shared_image_and_preserves_original_columns(self):
        frame=pd.DataFrame([dict(parent_asin='A',image_url='https://m.media-amazon.com/a.jpg',color='gold',pattern='plain'),
                            dict(parent_asin='B',image_url='https://m.media-amazon.com/a.jpg',color='maroon',pattern='plain'),
                            dict(parent_asin='C',image_url=None,color='blue',pattern='plain')])
        products,images=prepare_input(frame)
        record={**images[0],'label':'Shirt','score':.95,'margin':.5}
        with tempfile.TemporaryDirectory() as directory:
            report=export_results(products,{record['image_key']:record},directory)
            result=pd.read_parquet(Path(directory)/'products_annotated.parquet')
            self.assertEqual(result.parent_asin.tolist(),['A','B','C'])
            self.assertEqual(result.color.tolist(),['gold','maroon','blue'])
            self.assertEqual(result.garment_status.tolist(),['shirt_candidate','shirt_candidate','download_failed'])
            self.assertEqual(report['products'],3)
            self.assertEqual(len(pd.read_parquet(Path(directory)/'shirt_candidates.parquet')),2)
            self.assertEqual(len(pd.read_parquet(Path(directory)/'uncertain_products.parquet')),1)

    def test_evaluation_measures_retention_and_false_acceptance_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            pd.DataFrame([dict(parent_asin='A',reference_label='Shirt'),
                          dict(parent_asin='B',reference_label='Shirt'),
                          dict(parent_asin='C',reference_label='Blazer')]).to_csv(path/'labels.csv',index=False)
            pd.DataFrame([dict(parent_asin='A',garment_type='Shirt',garment_status='shirt_candidate'),
                          dict(parent_asin='B',garment_type='Shirt',garment_status='uncertain'),
                          dict(parent_asin='C',garment_type='Blazer',garment_status='non_shirt_candidate')]).to_parquet(path/'predictions.parquet',index=False)
            result=evaluate_labels(path/'labels.csv',path/'predictions.parquet')
            self.assertEqual(result['top1_accuracy'],1.)
            self.assertEqual(result['accepted_shirt_precision'],1.)
            self.assertEqual(result['genuine_shirt_retention'],.5)
            self.assertEqual(result['non_shirt_false_accept_rate'],0.)


if __name__=='__main__':unittest.main()
