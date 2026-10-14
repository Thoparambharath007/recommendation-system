"""FashionSigLIP prompt classification; preserve all original product rows."""
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
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import pandas as pd
import requests
from PIL import Image, ImageOps
from garment_core import VERSION, MODEL, REVISION, LABELS, Checkpoint, prepare_input, export_results, evaluate_labels

DESCRIPTIONS={
 'Shirt':['a button down shirt with a collar and sleeves','a folded dress shirt','a standalone casual collared shirt'],
 'T-shirt':['a crew neck t shirt','a casual tee shirt','a round neck short sleeve t shirt'],
 'Polo':['a polo shirt with a short button placket','a collared knitted polo shirt','a short sleeve polo top'],
 'Blazer':['a suit blazer with lapels','a tailored business suit jacket','a formal blazer jacket'],
 'Coat':['a long overcoat','a winter coat','a trench coat'],
 'Denim Jacket':['a denim jacket with chest pockets','a jean jacket','a heavy denim trucker jacket'],
 'Dresses':['a dress','a long dress','a short dress'],
 'Hoodie':['a hooded sweatshirt','a hoodie','a sweatshirt with a hood'],
 'Jacket':['an outerwear jacket','a zip up jacket','a casual jacket'],
 'Jeans':['a pair of denim jeans','blue jeans pants','denim trousers'],
 'Long Pants':['a pair of long trousers','a pair of pants','full length pants'],
 'Shorts':['a pair of shorts','short pants','casual shorts'],
 'Skirt':['a skirt','a short skirt','a long skirt'],
 'Sports Jacket':['a sports track jacket','an athletic zip up jacket','a training jacket'],
 'Sweater':['a knitted sweater','a pullover sweater','a wool jumper'],
 'Bodysuit':['a one piece dance bodysuit with a shirt top and crotch fastening','a long sleeve leotard','a shirt style bodysuit extending below the waist'],
 'Romper':['a one piece romper','a short leg playsuit','a one piece outfit with short legs'],
 'Vest':['a sleeveless waistcoat','a formal suit vest','a sleeveless vest'],
 'Jumpsuit':['a one piece jumpsuit','a full length coverall','a one piece outfit with long legs'],
}
PROMPTS={label:['a product photo of '+text for text in DESCRIPTIONS[label]] for label in LABELS}


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
        import open_clip
        self.torch=torch;self.device=device
        snapshot=Path(snapshot)
        config=json.loads((snapshot/'open_clip_config.json').read_text())
        # Use the pinned local checkpoint; no remote Python model code executes.
        config['model_cfg']['text_cfg']['hf_tokenizer_name']=str(snapshot)
        registry=snapshot/'local_configs';registry.mkdir(exist_ok=True)
        (registry/'ShirtFashionAudit.json').write_text(json.dumps(config['model_cfg']))
        open_clip.add_model_config(str(registry))
        pp=config['preprocess_cfg']
        self.model,_,self.processor=open_clip.create_model_and_transforms(
            'ShirtFashionAudit',pretrained=str(snapshot/'open_clip_model.safetensors'),device=device,
            image_mean=pp['mean'],image_std=pp['std'],image_interpolation=pp['interpolation'],image_resize_mode=pp['resize_mode'])
        self.model.eval();self.labels=LABELS
        tokenizer=open_clip.get_tokenizer('ShirtFashionAudit')
        tokens=tokenizer([p for label in LABELS for p in PROMPTS[label]]).to(device)
        with torch.inference_mode():
            features=self.model.encode_text(tokens,normalize=True).float().reshape(len(LABELS),3,-1).mean(1)
            self.text_features=torch.nn.functional.normalize(features,dim=-1)
    def predict(self,items):
        torch=self.torch
        records,images=zip(*items)
        inputs=torch.stack([self.processor(image) for image in images]).to(self.device)
        context=torch.autocast('cuda',dtype=torch.float16) if self.device!='cpu' else nullcontext()
        with torch.inference_mode(),context:
            features=self.model.encode_image(inputs,normalize=True).float()
            # Relative prompt scores, NOT calibrated probabilities of correctness.
            similarities=features @ self.text_features.T
            probs=(100.0*similarities).softmax(-1).cpu()
        values,indices=probs.topk(3,dim=-1)
        results=[]
        for row,record in enumerate(records):
            result={**record,'label':self.labels[int(indices[row,0])],'score':float(values[row,0]),
                    'margin':float(values[row,0]-values[row,1]),
                    'scores_json':json.dumps({label:float(score) for label,score in zip(self.labels,probs[row])}),
                    'top3_json':json.dumps([{'label':self.labels[int(i)],'score':float(v)} for i,v in zip(indices[row],values[row])])}
            results.append(result)
        return [{**result,'score_kind':'relative_prompt_softmax_not_accuracy'} for result in results]


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
                    allow_patterns=['open_clip_config.json','open_clip_model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','spiece.model'])
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
