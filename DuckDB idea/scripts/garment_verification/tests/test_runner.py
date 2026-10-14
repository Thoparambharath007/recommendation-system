import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classify_garments import run


class FakeClassifier:
    def __init__(self,*args):pass
    def predict(self,items):
        return [{**record,'label':'Shirt','score':.95,'margin':.6} for record,image in items]


class RunnerTests(unittest.TestCase):
    def test_parallel_batch_mapping_failures_and_resume_without_model_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            path=base/'input.csv'
            pd.DataFrame([dict(parent_asin=asin,image_url=url) for asin,url in [
                ('A','https://example.com/a.jpg'),('B','https://example.com/a.jpg'),
                ('C','https://example.com/fail.jpg'),('D','https://example.com/d.jpg')]]).to_csv(path,index=False)
            args=argparse.Namespace(input=str(path),output_dir=str(base/'out'),devices='0,1',batch_size=2,
                                    download_workers=2,limit_images=0,preview_count=0,min_score=.85,min_margin=.2,retry_failed=False)
            def download(item,*args):
                if 'fail' in item['image_url']:return {**item,'error':'HTTP 404'},None
                return item,Image.new('RGB',(8,8),'white')
            with patch('huggingface_hub.snapshot_download',return_value='fake') as snapshot, \
                 patch('classify_garments.Classifier',FakeClassifier),patch('classify_garments.download_image',side_effect=download) as fetch, \
                 patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.device_count',return_value=2),redirect_stdout(io.StringIO()):
                run(args)
                self.assertEqual(fetch.call_count,3)
                run(args)
                self.assertEqual(fetch.call_count,3)
                self.assertEqual(snapshot.call_count,1)
            result=pd.read_parquet(base/'out/products_annotated.parquet')
            self.assertEqual(result.garment_status.tolist(),['shirt_candidate','shirt_candidate','download_failed','shirt_candidate'])


if __name__=='__main__':unittest.main()
