"""Build the reviewed Kaggle archive without modifying the master catalog."""
from pathlib import Path
import json
import hashlib
import zipfile

HERE=Path(__file__).resolve().parent
MASTER=HERE.parents[1]/'data/shirts_master.parquet'

def build():
    digest=hashlib.sha256(MASTER.read_bytes()).hexdigest()
    cells=[]
    def cell(kind,text):
        entry={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':entry.update(execution_count=None,outputs=[])
        cells.append(entry)
    cell('markdown','# Garment verification v2 — full catalog\nEnable Internet and GPU T4 x2. Attach the v2 ZIP. This processes unique URLs and preserves original IDs. No more mandatory pilot rounds. Relative prompt scores are not accuracy guarantees.\n')
    cell('code',r'''from pathlib import Path
import os, sys, subprocess, zipfile, shutil, json, hashlib
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
unpack=Path('/kaggle/working/garment_code_v2')
candidates=[p for p in Path('/kaggle/input').rglob('garment_core.py') if "VERSION='2.0.0'" in p.read_text()]
if not candidates:
    archives=list(Path('/kaggle/input').rglob('shirt_garment_verification_v2.zip'))
    assert len(archives)==1, 'Attach the v2 ZIP.'
    unpack.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as z:
        assert all((unpack/n).resolve().is_relative_to(unpack.resolve()) for n in z.namelist())
        z.extractall(unpack)
    candidates=list(unpack.rglob('garment_core.py'))
assert len(candidates)==1, f'Select one v2 code package: {candidates}'
CODE=candidates[0].parent
DATA=CODE/'input/shirts_master.parquet'
SCRIPT=CODE/'classify_garments.py'
OUT=Path('/kaggle/working/garment_results_v2')
manifest=json.loads((CODE/'package_manifest.json').read_text())
assert hashlib.sha256(DATA.read_bytes()).hexdigest()==manifest['master_sha256']
RESTORE_ZIP=None # Optional Path('/kaggle/input/.../garment_results_v2_backup.zip')
if RESTORE_ZIP:
    assert not (OUT/'checkpoint.sqlite').exists(), 'Existing checkpoint: do not overwrite it.'
    with zipfile.ZipFile(RESTORE_ZIP) as z:
        base=OUT.parent.resolve()
        assert all((base/n).resolve().is_relative_to(base) for n in z.namelist())
        assert all(n.startswith(OUT.name+'/') for n in z.namelist())
        z.extractall(base)
DEPS=Path('/kaggle/working/garment_deps_v2')
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--upgrade','--target',str(DEPS),'-r',str(CODE/'requirements-kaggle.txt')],check=True)
ENV=os.environ.copy()
ENV['PYTHONPATH']=str(DEPS)+os.pathsep+ENV.get('PYTHONPATH','')
print('Code:',CODE,'Data:',DATA,'Results:',OUT,sep='\n')
''')
    cell('code',r'''import torch
assert torch.cuda.is_available() and torch.cuda.device_count()>=2, 'Select GPU T4 x2.'
print(torch.__version__,[torch.cuda.get_device_name(i) for i in range(2)])
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(CODE/'tests'),'-v'],env=ENV,check=True)
''')
    cell('markdown','## Full catalog\nRun this cell. LIMIT=0 processes every remaining unique URL. LIMIT=12 is available for a short optional speed check; change it back to 0 to finish. Existing v2 checkpoint results resume automatically.\n')
    cell('code',r'''LIMIT=0
subprocess.run([sys.executable,'-u',str(SCRIPT),'run','--input',str(DATA),
    '--output-dir',str(OUT),'--devices','0,1','--batch-size','24',
    '--download-workers','12','--min-score','0.85','--min-margin','0.20',
    '--limit-images',str(LIMIT),'--preview-count','100'],env=ENV,check=True)
print((OUT/'summary.json').read_text())
''')
    cell('markdown','## Download results\nAfter the worker exits, run this cell and download the linked ZIP. Keep uncertain products separate; do not replace the master blindly.\n')
    cell('code',r'''from IPython.display import FileLink, display
assert OUT.exists()
archive=shutil.make_archive('/kaggle/working/garment_results_v2_backup','zip',OUT.parent,OUT.name)
display(FileLink(archive))
''')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    (HERE/'Kaggle_Start.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    (HERE/'package_manifest.json').write_text(json.dumps({'version':'2.0.0','master_sha256':digest,'master_products':50224,'unique_urls':31793},indent=2))
    archive=HERE/'shirt_garment_verification_v2.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for n in ['garment_core.py','classify_garments.py','catalog_quality.py','requirements-kaggle.txt','README.md','Kaggle_Start.ipynb','package_manifest.json','VALIDATION.json']:
            z.write(HERE/n,'shirt_garment_verification_v2/'+n)
        for folder in ['sample','tests']:
            for p in (HERE/folder).glob('*'):
                if p.is_file():z.write(p,'shirt_garment_verification_v2/'+p.relative_to(HERE).as_posix())
        z.write(MASTER,'shirt_garment_verification_v2/input/shirts_master.parquet')
    assert hashlib.sha256(MASTER.read_bytes()).hexdigest()==digest
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    for c in cells:
        if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
    print(archive.resolve())

if __name__=='__main__':build()
