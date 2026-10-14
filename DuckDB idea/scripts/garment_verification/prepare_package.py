"""Prepare diverse unique samples and the exact Kaggle upload archive."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import pandas as pd
from garment_core import VERSION, MODEL, REVISION, LABELS, prepare_input

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MASTER=ROOT/'data/shirts_master.parquet'


def make_sample(master,count=300):
    products,images=prepare_input(master)
    unique=products.dropna(subset=['garment_image_key']).drop_duplicates('garment_image_key').copy()
    unique=unique.sample(frac=1,random_state=42)
    selected={}
    def take(frame,group,n):
        if n<=0 or len(selected)>=count:return
        added=0
        for row in frame.to_dict('records'):
            key=row['garment_image_key']
            if key not in selected:
                selected[key]={**row,'sample_group':group};added+=1
            if added>=n or len(selected)>=count:break
    known=['B076PQFJRG','B00ZZ4D8H8','B09ZJ5QKQN','B07T3G62SP','B077LY6DM2','B00S0LWN0Q','B00Y2A3OAW','B0BJFDV3CG','B0B9BQ867Q','B09HMXGBHQ']
    take(unique[unique.parent_asin.isin(known)],'previously_reported_example',len(known))
    suspicious=unique.title.fillna('').str.contains(r'\b(?:blazer|jackets?|t[ -]?shirts?|hoodie|polo|coat|vest)\b',case=False,regex=True)
    take(unique[suspicious],'suspected_other_type_not_verified',40)
    for (sleeve,pattern),group in unique.groupby(['sleeve','pattern'],dropna=False,sort=True):
        take(group,f'sleeve={sleeve};pattern={pattern}',4)
    for color,group in unique.groupby('color_main',dropna=False,sort=True):take(group,'main_color='+str(color),2)
    for status,group in unique.groupby('vision_status',sort=True):take(group,'previous_vision_status='+str(status),4)
    take(unique,'additional_random_examples',count-len(selected))
    sample=pd.DataFrame(selected.values()).head(count)
    fields=[c for c in ['parent_asin','title','image_url','sleeve','pattern','color','color_main','brand','vision_status','sample_group'] if c in sample]
    sample=sample[fields]
    assert sample.image_url.nunique()==len(sample)
    return sample


def notebook():
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':text.splitlines(keepends=True)})
    md('# Shirt / T-shirt / blazer verification — Dima806 only\n\nSelect **GPU T4 ×2** and enable Internet. Attach `shirt_garment_verification_v1.zip` as an input dataset. This classifies images; it does not train a model or replace your project database.\n\nRun setup, then the 300-image sample. The full-catalog cell is explicitly enabled when you choose. Save the result ZIP before ending the session.')
    code('''from pathlib import Path
import sys, os, subprocess, json, shutil, zipfile, hashlib
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
BASE=Path('/kaggle/input')
candidates=[p for p in BASE.rglob('classify_garments.py') if (p.parent/'garment_core.py').exists()]
if not candidates:
    archives=list(BASE.rglob('shirt_garment_verification_v1.zip'))
    assert len(archives)==1, f'Attach one supplied input ZIP: {archives}'
    unpack=Path('/kaggle/working/garment_code_v1')
    unpack.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as z:
        assert all((unpack/n).resolve().is_relative_to(unpack.resolve()) for n in z.namelist())
        z.extractall(unpack)
    candidates=list(unpack.rglob('classify_garments.py'))
assert len(candidates)==1, f'Attach one version of this package: {candidates}'
CODE=candidates[0].parent
SCRIPT=CODE/'classify_garments.py'
DATA=CODE/'input/shirts_master.parquet'
SAMPLE=CODE/'sample/sample_input.csv'
OUT=Path('/kaggle/working/garment_results_v1')
assert DATA.exists() and SAMPLE.exists(), 'The package must include input/ and sample/.'
manifest=json.loads((CODE/'package_manifest.json').read_text())
assert hashlib.sha256(DATA.read_bytes()).hexdigest()==manifest['master_sha256'], 'Input master checksum mismatch'
print('Code:',CODE,'\\nMaster:',DATA,'\\nSample:',SAMPLE,'\\nResults:',OUT)
DEPS=Path('/kaggle/working/garment_python_v1')
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','--upgrade','--target',str(DEPS),
                '-r',str(CODE/'requirements-kaggle.txt')],check=True)
os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+os.environ.get('PYTHONPATH','')
# Worker subprocesses use this isolated overlay. Kaggle's global Hub/Gradio and
# CUDA PyTorch installation are not downgraded or replaced.
''')
    md('## Optional restore — run before processing images\nFor a new session, attach the earlier output ZIP and set its path here. Leave None for your first run. Restore before running sample/full cells; never overwrite an existing checkpoint.')
    code('''RESTORE_ZIP=None  # Path('/kaggle/input/.../garment_results_backup.zip')
if RESTORE_ZIP:
    assert RESTORE_ZIP.exists()
    assert not (OUT/'checkpoint.sqlite').exists(), 'Output already has a checkpoint; do not overwrite it.'
    with zipfile.ZipFile(RESTORE_ZIP) as z:
        assert all((OUT.parent/n).resolve().is_relative_to(OUT.parent.resolve()) for n in z.namelist())
        z.extractall(OUT.parent)
    assert (OUT/'checkpoint.sqlite').exists(), 'Backup output folder name differs; select its OUT path.'
''')
    md('## Confirm hardware and run package checks\nDo not install a replacement PyTorch wheel. Kaggle supplies CUDA-enabled PyTorch.')
    code('''import torch
assert torch.cuda.is_available() and torch.cuda.device_count()>=2, 'Select GPU T4 x2 in Kaggle Settings.'
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(2)])
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(CODE/'tests'),'-v'],check=True)
''')
    md('## Run the varied sample\n`LIMIT=0` means all remaining sample images. It does **not** mean the full catalog. Scores/margins are heuristic gates, not measured accuracy. Shirt is the only accepted target; T-shirt and Polo remain separate classes.')
    code('''LIMIT=0
MIN_SCORE=0.85
MIN_MARGIN=0.20
COMMON=['--output-dir',str(OUT),'--devices','0,1','--batch-size','24',
        '--download-workers','12','--min-score',str(MIN_SCORE),'--min-margin',str(MIN_MARGIN)]
subprocess.run([sys.executable,'-u',str(SCRIPT),'run','--input',str(SAMPLE),
                '--limit-images',str(LIMIT),'--preview-count','300',*COMMON],check=True)
''')
    code('''import pandas as pd
from IPython.display import display, Image as DisplayImage
summary=json.loads((OUT/'summary.json').read_text())
display(summary)
results=pd.read_parquet(OUT/'products_annotated.parquet')
display(results[['parent_asin','title','sleeve','pattern','garment_type','garment_score','garment_margin','garment_status']].head(30))
for path in sorted((OUT/'previews').glob('*.jpg'))[:12]:
    print(path.stem)
    display(DisplayImage(filename=str(path)))
speed=summary.get('images_per_second')
if speed:
    remaining=max(0,manifest['unique_valid_urls']-summary['processed_unique_urls'])
    print(f'Illustrative remaining processing time at this measured sample rate: {remaining/speed/3600:.2f} h')
    print('Startup/download overhead and network variation affect this estimate.')
''')
    md('## Optional measured evaluation\nDownload `reference_labels.csv`, fill a small verified set using the exact category names in `label_names.json`, then attach that CSV as a new input. The script reports shirt retention and non-shirt false acceptance. Leave this cell disabled if no reference labels have been supplied; do not interpret model scores as accuracy.')
    code('''RUN_EVALUATION=False
LABELS=None  # Set Path('/kaggle/input/.../your_verified_labels.csv') if available.
if RUN_EVALUATION:
    assert LABELS and LABELS.exists(), 'Select your manually verified label CSV.'
    subprocess.run([sys.executable,str(SCRIPT),'evaluate','--labels',str(LABELS),
                    '--predictions',str(OUT/'products_annotated.parquet'),
                    '--output',str(OUT/'evaluation.json')],check=True)
    display(json.loads((OUT/'evaluation.json').read_text()))
''')
    md('## Full catalog — reuse the sample checkpoint\nSet `RUN_FULL_DATASET=True` when ready. The input is the original 50,224-row master, processed once per unique URL. Existing sample predictions resume automatically. No model training is performed.\n\nFor a resumed session, use the restore cell near the top before processing. You can then skip the sample and run this full-catalog cell.')
    code('''RUN_FULL_DATASET=False
if RUN_FULL_DATASET:
    subprocess.run([sys.executable,'-u',str(SCRIPT),'run','--input',str(DATA),
                    '--preview-count','300',*COMMON],check=True)
    display(json.loads((OUT/'summary.json').read_text()))
''')
    md('## Save results\nRun after the worker exits. Download the entire ZIP, including the checkpoint and previews. `products_annotated.parquet` retains all input rows; the three candidate files partition those rows. None replaces your live database automatically. Failed/pending images stay in the uncertain output.')
    code('''from IPython.display import FileLink
assert OUT.exists()
archive=shutil.make_archive('/kaggle/working/garment_results_backup','zip',OUT.parent,OUT.name)
display(FileLink(archive))
''')
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'cells':cells}


def main():
    before=hashlib.sha256(MASTER.read_bytes()).hexdigest()
    frame=pd.read_parquet(MASTER)
    _,images=prepare_input(frame)
    sample=make_sample(frame)
    sample_dir=HERE/'sample';sample_dir.mkdir(exist_ok=True)
    sample.to_csv(sample_dir/'sample_input.csv',index=False)
    labels=sample[['parent_asin','image_url','sample_group']].copy()
    labels['reference_label']='';labels['notes']='';labels.to_csv(sample_dir/'reference_labels.csv',index=False)
    (HERE/'label_names.json').write_text(json.dumps(LABELS,indent=2),encoding='utf-8')
    manifest={'package_version':VERSION,'model':MODEL,'revision':REVISION,'master_sha256':before,
              'master_products':len(frame),'unique_valid_urls':len(images),'sample_images':len(sample),
              'sample_sleeves':sample.sleeve.value_counts().to_dict(),
              'sample_patterns':sample.pattern.value_counts().to_dict(),
              'sample_main_colors':sample.color_main.value_counts().to_dict(),
              'note':'Sample groups come from existing metadata, not verified garment labels.'}
    (HERE/'package_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (HERE/'Kaggle_Start.ipynb').write_text(json.dumps(notebook(),indent=1),encoding='utf-8')
    archive=HERE/'shirt_garment_verification_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        names=['garment_core.py','classify_garments.py','requirements-kaggle.txt','README.md','VALIDATION.md','Kaggle_Start.ipynb','package_manifest.json','label_names.json']
        for name in names:z.write(HERE/name,'shirt_garment_verification_v1/'+name)
        for folder in ['tests','sample']:
            for path in sorted((HERE/folder).glob('*')):
                if path.is_file():z.write(path,'shirt_garment_verification_v1/'+path.relative_to(HERE).as_posix())
        z.write(MASTER,'shirt_garment_verification_v1/input/shirts_master.parquet')
    assert hashlib.sha256(MASTER.read_bytes()).hexdigest()==before
    print(json.dumps(manifest,indent=2));print('Upload ZIP:',archive);print('Import notebook:',HERE/'Kaggle_Start.ipynb')


if __name__=='__main__':main()
