"""Input identity, resumable raw predictions, and non-destructive exports."""
import hashlib
import html
import json
import math
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit
import pandas as pd

VERSION='1.0.0'
MODEL='dima806/clothes_image_detection'
REVISION='649dbb348805d6b711a6da28dc58531259280b77'
LABELS=['Blazer','Coat','Denim Jacket','Dresses','Hoodie','Jacket','Jeans','Long Pants','Polo','Shirt','Shorts','Skirt','Sports Jacket','Sweater','T-shirt']


def image_key(value):
    if not isinstance(value,str):return None
    value=value.strip()
    try:
        url=urlsplit(value)
        if url.scheme not in {'https','http'} or not url.hostname:return None
    except ValueError:return None
    return hashlib.sha256(value.encode()).hexdigest()


def prepare_input(frame):
    if not {'parent_asin','image_url'}<=set(frame):raise ValueError('Input requires parent_asin and image_url')
    if frame.parent_asin.isna().any():raise ValueError('Missing product ID')
    ids=frame.parent_asin.astype(str).str.strip()
    if ids.eq('').any() or ids.duplicated().any():raise ValueError('Product IDs must be unique and nonempty')
    reserved={'garment_image_key','garment_type','garment_score','garment_margin','garment_status','garment_reason','garment_model'}
    if reserved & set(frame):raise ValueError('Use the original input, not an already annotated output')
    products=frame.copy(deep=True)
    products['garment_image_key']=pd.Series([image_key(url) for url in products.image_url],index=products.index,dtype=object)
    seen={}
    for row in products[['image_url','garment_image_key']].itertuples(index=False):
        if row.garment_image_key:seen.setdefault(row.garment_image_key,{'image_key':row.garment_image_key,'image_url':row.image_url.strip()})
    return products,list(seen.values())


def decide(record,min_score=.85,min_margin=.20):
    if record.get('error'):return 'download_failed',str(record['error'])[:300]
    if not record.get('label'):return 'not_processed','Image has not been processed'
    score,margin=record.get('score'),record.get('margin')
    if record['label'] not in LABELS or not all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in [score,margin]):
        return 'uncertain','Invalid classifier output'
    if score<min_score or margin<min_margin:return 'uncertain','Top score or runner-up separation is below the heuristic threshold'
    return ('shirt_candidate' if record['label']=='Shirt' else 'non_shirt_candidate'), 'Strong model prediction; not human verified'


class Checkpoint:
    def __init__(self,directory,identity):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.con=sqlite3.connect(self.directory/'checkpoint.sqlite',timeout=60)
        self.con.execute('PRAGMA journal_mode=WAL')
        self.con.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.con.execute('CREATE TABLE IF NOT EXISTS predictions (image_key TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        encoded=json.dumps(identity,sort_keys=True)
        row=self.con.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if row and row[0]!=encoded:
            self.con.close()
            raise ValueError('Model/code identity changed. Use a new output folder; do not overwrite the old checkpoint.')
        self.con.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)",[encoded]);self.con.commit()
    def keys(self):return {row[0] for row in self.con.execute('SELECT image_key FROM predictions')}
    def results(self):return {key:json.loads(payload) for key,payload in self.con.execute('SELECT image_key,payload FROM predictions')}
    def put_many(self,records):
        with self.con:
            self.con.executemany('INSERT OR REPLACE INTO predictions VALUES (?,?)',
                                 [(r['image_key'],json.dumps(r,allow_nan=False)) for r in records])
    def retry_failed(self):
        keys=[key for key,r in self.results().items() if r.get('error')]
        with self.con:self.con.executemany('DELETE FROM predictions WHERE image_key=?',[(k,) for k in keys])
    def __enter__(self):return self
    def __exit__(self,*args):
        self.con.execute('PRAGMA wal_checkpoint(TRUNCATE)');self.con.close()


def atomic_parquet(frame,path):
    path=Path(path);tmp=path.with_suffix('.parquet.tmp')
    frame.to_parquet(tmp,index=False);tmp.replace(path)


def export_results(products,records,directory,min_score=.85,min_margin=.20):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    output=products.copy(deep=True)
    details=[]
    for key in output.garment_image_key:
        record=records.get(key,{}) if key else {'error':'Missing or invalid image URL'}
        status,reason=decide(record,min_score,min_margin)
        details.append({'garment_type':record.get('label'),'garment_score':record.get('score'),
                        'garment_margin':record.get('margin'),'garment_status':status,'garment_reason':reason,
                        'garment_model':MODEL+'@'+REVISION})
    for column in details[0] if details else ['garment_type','garment_score','garment_margin','garment_status','garment_reason','garment_model']:
        output[column]=[d[column] for d in details]
    atomic_parquet(output,directory/'products_annotated.parquet')
    for filename,statuses in [('shirt_candidates.parquet',['shirt_candidate']),
                              ('non_shirt_candidates.parquet',['non_shirt_candidate']),
                              ('uncertain_products.parquet',['uncertain','download_failed','not_processed'])]:
        atomic_parquet(output[output.garment_status.isin(statuses)],directory/filename)
    columns=['parent_asin','image_url','garment_type','garment_score','garment_margin','garment_status','garment_reason']
    if 'sample_group' in output:columns.append('sample_group')
    output[columns].to_csv(directory/'product_verdicts.csv',index=False)
    unique_keys=set(output.garment_image_key.dropna())
    images=pd.DataFrame([records[k] for k in sorted(unique_keys) if k in records])
    if images.empty:images=pd.DataFrame(columns=['image_key','image_url','label','score','margin','error'])
    atomic_parquet(images,directory/'image_predictions.parquet')
    images.to_csv(directory/'image_predictions.csv',index=False)
    labels=output[['parent_asin','image_url','garment_type']].drop_duplicates('image_url').copy()
    labels['reference_label']='';labels['notes']=''
    label_path=directory/'reference_labels.csv'
    if not label_path.exists():labels.to_csv(label_path,index=False)
    counts=output.garment_status.value_counts().to_dict()
    report={'products':len(output),'unique_valid_urls':len(unique_keys),'processed_unique_urls':len(unique_keys & set(records)),
            'pending_unique_urls':len(unique_keys-set(records)), 'product_status_counts':counts,
            'minimum_score':min_score,'minimum_margin':min_margin,'model':MODEL,'revision':REVISION,
            'note':'Candidates are model predictions, not ground truth. T-shirts and polos are separate from Shirt. Original input is unchanged.'}
    (directory/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cards=[]
    for row in output.head(400).to_dict('records'):
        key=row['garment_image_key']
        preview=directory/'previews'/f'{key}.jpg'
        src=f'previews/{key}.jpg' if preview.is_file() else str(row.get('image_url') or '')
        if not image_key(src) and not preview.is_file():src=''
        title=html.escape(str(row.get('title') or row['parent_asin']))
        cards.append(f'<article><h3>{html.escape(str(row["parent_asin"]))} — {title}</h3>'
                     f'<img loading="lazy" src="{html.escape(src,quote=True)}" alt="Product image">'
                     f'<p>{html.escape(str(row["garment_type"]))} · score {row["garment_score"]} · {row["garment_status"]}</p>'
                     f'<p>{html.escape(str(row.get("sample_group", "")))}</p></article>')
    (directory/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Garment predictions</title>'
        '<style>body{font:16px Arial;max-width:1100px;margin:24px auto;background:#f5f5f5}article{background:white;padding:16px;margin:16px}img{max-width:360px;max-height:360px;object-fit:contain}</style>'
        '<h1>Garment predictions</h1><p>Review sample results. Scores are not accuracy guarantees. Open this file together with the previews folder.</p>'
        +''.join(cards),encoding='utf-8')
    return report


def evaluate_labels(label_path,predictions_path):
    labels=pd.read_csv(label_path,keep_default_na=False)
    labels=labels[labels.reference_label.str.strip().ne('')]
    if labels.empty:raise ValueError('No verified reference labels have been filled in')
    if labels.parent_asin.duplicated().any() or not set(labels.reference_label)<=set(LABELS):raise ValueError('Use unique IDs and exact model label names')
    predictions=pd.read_parquet(predictions_path)
    joined=labels.merge(predictions[['parent_asin','garment_type','garment_status']],on='parent_asin',how='left',validate='one_to_one',indicator=True)
    if joined['_merge'].ne('both').any():raise ValueError('Reference IDs are missing from predictions')
    shirt=joined.reference_label.eq('Shirt');accepted=joined.garment_status.eq('shirt_candidate')
    def ratio(n,d):return float(n/d) if d else None
    return {'verified_images':len(joined),'top1_accuracy':float(joined.reference_label.eq(joined.garment_type).mean()),
            'accepted_shirt_precision':ratio((shirt&accepted).sum(),accepted.sum()),
            'genuine_shirt_retention':ratio((shirt&accepted).sum(),shirt.sum()),
            'non_shirt_false_accept_rate':ratio((~shirt&accepted).sum(),(~shirt).sum()),
            'confusion_counts':pd.crosstab(joined.reference_label,joined.garment_type.fillna('missing')).to_dict(),
            'note':'Valid only for this labeled sample. Small or selectively chosen samples do not establish whole-catalog accuracy.'}
