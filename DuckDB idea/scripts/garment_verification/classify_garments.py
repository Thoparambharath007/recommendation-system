"""Dima806 batch inference; run on Kaggle, preserve the input catalog."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import pandas as pd
import requests
from PIL import Image, ImageOps
from garment_core import VERSION, MODEL, REVISION, LABELS, Checkpoint, prepare_input, export_results, evaluate_labels


def download_image(item,preview_dir=None):
    record=dict(item)
    for attempt in range(2):
        try:
            with requests.get(item['image_url'],timeout=(8,20),stream=True,
                              headers={'User-Agent':'Mozilla/5.0 ShirtResearch/1.0'}) as response:
                response.raise_for_status()
                chunks=[];size=0
                for chunk in response.iter_content(65536):
                    size+=len(chunk)
                    if size>12*1024*1024:raise ValueError('Image exceeds 12 MB')
                    chunks.append(chunk)
            blob=b''.join(chunks)
            record['image_sha256']=hashlib.sha256(blob).hexdigest()
            with Image.open(io.BytesIO(blob)) as source:
                if source.width*source.height>30_000_000:raise ValueError('Image dimensions are too large')
                image=ImageOps.exif_transpose(source).convert('RGB')
                record['image_width'],record['image_height']=image.size
                image.thumbnail((768,768))
            if preview_dir is not None:
                preview_dir.mkdir(parents=True,exist_ok=True)
                image.save(preview_dir/(item['image_key']+'.jpg'),quality=85)
            record.pop('error',None)
            return record,image
        except (requests.RequestException,OSError,ValueError,Image.DecompressionBombError) as exc:
            record['error']=type(exc).__name__+': '+str(exc)[:220]
            if attempt==0:time.sleep(.5)
    return record,None


class Classifier:
    def __init__(self,snapshot,device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.torch=torch;self.device=device
        self.processor=AutoImageProcessor.from_pretrained(snapshot,local_files_only=True)
        self.model=AutoModelForImageClassification.from_pretrained(snapshot,local_files_only=True,use_safetensors=True).to(device).eval()
        self.labels=[self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        if self.labels!=LABELS:raise ValueError('Unexpected model label mapping; refusing to guess class IDs')
    def predict(self,items):
        torch=self.torch
        records,images=zip(*items)
        inputs=self.processor(images=list(images),return_tensors='pt').to(self.device)
        context=torch.autocast('cuda',dtype=torch.float16) if self.device!='cpu' else nullcontext()
        with torch.inference_mode(),context:
            probs=self.model(**inputs).logits.float().softmax(-1).cpu()
        values,indices=probs.topk(3,dim=-1)
        results=[]
        for row,record in enumerate(records):
            result={**record,'label':self.labels[int(indices[row,0])],'score':float(values[row,0]),
                    'margin':float(values[row,0]-values[row,1]),
                    'scores_json':json.dumps({label:float(score) for label,score in zip(self.labels,probs[row])}),
                    'top3_json':json.dumps([{'label':self.labels[int(i)],'score':float(v)} for i,v in zip(indices[row],values[row])])}
            results.append(result)
        return results


def load_input(path):
    path=Path(path)
    return pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path,keep_default_na=False,dtype={'parent_asin':str})


def run(args):
    if args.batch_size<1 or args.download_workers<1 or args.limit_images<0:raise ValueError('Invalid batch/download/limit settings')
    if not 0<=args.min_score<=1 or not 0<=args.min_margin<=1:raise ValueError('Thresholds must be between 0 and 1')
    directory=Path(args.output_dir).resolve()
    if directory==Path(args.input).resolve().parent:raise ValueError('Use a separate output directory, not the input folder')
    products,images=prepare_input(load_input(args.input))
    source_dir=Path(__file__).resolve().parent
    digest=hashlib.sha256((source_dir/'garment_core.py').read_bytes()+(source_dir/'classify_garments.py').read_bytes()).hexdigest()
    identity={'version':VERSION,'model':MODEL,'revision':REVISION,'code_sha256':digest,'preprocess_max_side':768}
    started=time.perf_counter();processed=0
    with Checkpoint(directory,identity) as state:
        if args.retry_failed:state.retry_failed()
        completed=state.keys()
        pending=[image for image in images if image['image_key'] not in completed]
        if args.limit_images:pending=pending[:args.limit_images]
        print(f'{len(products):,} products; {len(images):,} unique URLs; {len(completed & {i["image_key"] for i in images}):,} already checkpointed; {len(pending):,} new images this run.',flush=True)
        (directory/'run_config.json').write_text(json.dumps({**identity,'input':str(Path(args.input).resolve()),
            'min_score':args.min_score,'min_margin':args.min_margin,'batch_size':args.batch_size,'devices':args.devices},indent=2),encoding='utf-8')
        try:
            if pending:
                import torch
                from huggingface_hub import snapshot_download
                torch.set_num_threads(2)
                devices=['cpu'] if args.devices=='cpu' else ['cuda:'+x.strip() for x in args.devices.split(',')]
                if len(set(devices))!=len(devices):raise ValueError('Device IDs must be distinct')
                for device in devices:
                    if device!='cpu' and (not torch.cuda.is_available() or int(device.split(':')[1])>=torch.cuda.device_count()):
                        raise ValueError('Requested GPU is unavailable; select Kaggle T4 x2')
                print('Downloading pinned classifier weights (once)...',flush=True)
                model_dir=Path(os.environ.get('GARMENT_MODEL_DIR',str(directory.parent/'garment_models'/REVISION)))
                snapshot=snapshot_download(MODEL,revision=REVISION,local_dir=str(model_dir),
                    allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])
                models=[Classifier(snapshot,device) for device in devices]
                preview_keys={i['image_key'] for i in images[:args.preview_count]}
                step=args.batch_size*len(models)
                with ThreadPoolExecutor(max_workers=args.download_workers) as downloads,ThreadPoolExecutor(max_workers=len(models)) as inference:
                    for offset in range(0,len(pending),step):
                        block=pending[offset:offset+step]
                        futures=[downloads.submit(download_image,item,directory/'previews' if item['image_key'] in preview_keys else None) for item in block]
                        good=[];failed=[]
                        for future in as_completed(futures):
                            record,image=future.result()
                            (good if image is not None else failed).append((record,image) if image is not None else record)
                        state.put_many(failed)
                        jobs=[]
                        for index,model in enumerate(models):
                            batch=good[index::len(models)]
                            if batch:jobs.append(inference.submit(model.predict,batch))
                        for job in as_completed(jobs):state.put_many(job.result())
                        processed+=len(block)
                        elapsed=time.perf_counter()-started
                        speed=processed/elapsed if elapsed else 0
                        eta=(len(pending)-processed)/speed/3600 if speed else None
                        print(f'{processed:,}/{len(pending):,} new images | {speed:.2f} images/s combined | ETA {eta:.2f} h | download failures in batch: {len(failed)}',flush=True)
                        for record,image in good:image.close()
            records=state.results()
        finally:
            # Export even if inference is interrupted/fails: never call pending
            # or failed products non-shirts. SQLite already stores finished work.
            summary=export_results(products,state.results(),directory,args.min_score,args.min_margin)
            summary.update(new_images_this_run=processed,elapsed_seconds=time.perf_counter()-started,
                           images_per_second=processed/(time.perf_counter()-started) if processed else None)
            (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run_parser=sub.add_parser('run')
    run_parser.add_argument('--input',required=True);run_parser.add_argument('--output-dir',required=True)
    run_parser.add_argument('--devices',default='0,1');run_parser.add_argument('--batch-size',type=int,default=24)
    run_parser.add_argument('--download-workers',type=int,default=12)
    run_parser.add_argument('--limit-images',type=int,default=0,help='0 processes all remaining unique URLs')
    run_parser.add_argument('--preview-count',type=int,default=300)
    run_parser.add_argument('--min-score',type=float,default=.85);run_parser.add_argument('--min-margin',type=float,default=.20)
    run_parser.add_argument('--retry-failed',action='store_true')
    ev=sub.add_parser('evaluate');ev.add_argument('--labels',required=True);ev.add_argument('--predictions',required=True);ev.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='run':run(args)
    else:
        result=evaluate_labels(args.labels,args.predictions)
        Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
