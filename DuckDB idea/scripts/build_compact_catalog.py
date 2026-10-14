"""Build one shirt-candidate record per URL; never write the original master."""
from pathlib import Path
import sys,hashlib,json,zipfile,io
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from query_engine import FILTER_COLUMNS
from product_enrichment import SIZE_NAMES, BASE_SIZES

def compact_frames(frame):
    if frame.parent_asin.duplicated().any():raise ValueError('Product IDs must be unique')
    candidates=frame[frame.garment_status.eq('shirt_candidate')].copy()
    if candidates.image_url.isna().any() or candidates.image_url.str.strip().eq('').any():
        raise ValueError('Candidate photo URL is missing')
    candidates['image_url']=candidates.image_url.str.strip()
    # Deterministic representative: exact price first, then rating evidence.
    candidates['_exact']=candidates.price_source.fillna('').str.startswith('exact')
    candidates=candidates.sort_values(['_exact','rating_number','average_rating','parent_asin'],ascending=[False,False,False,True])
    products=candidates.drop_duplicates('image_url').drop(columns=['_exact']).copy()
    mapping=dict(zip(products.image_url,products.parent_asin))
    candidates['catalog_product_id']=candidates.image_url.map(mapping)
    # Size/listing metadata lives separately; photos and shared vision fields are
    # stored only in the compact main catalog, not repeated in the option table.
    fields=(FILTER_COLUMNS-{'color','color_main','pattern','vision_status','color_status','parent_asin'}) | {
        'parent_asin','catalog_product_id','size_source','size_evidence','size_note','brand_source'}
    options=candidates[[c for c in candidates.columns if c in fields]].copy()
    order={name:i for i,name in enumerate(BASE_SIZES)}
    sizes=candidates.groupby('catalog_product_id').sizes.apply(lambda values:'|'.join(sorted(
        {s for v in values.fillna('') for s in v.split('|') if s in SIZE_NAMES},key=lambda s:(order.get(s,100),s))))
    products['sizes']=products.parent_asin.map(sizes).fillna('')
    products['size_note']='Recorded size options across listings sharing this image; current stock is not verified.'
    products['catalog_option_count']=products.parent_asin.map(candidates.groupby('catalog_product_id').size())
    assert products.image_url.nunique()==len(products)
    return products.reset_index(drop=True),options.reset_index(drop=True)

def main():
    master=ROOT/'data/shirts_master.parquet'
    before=hashlib.sha256(master.read_bytes()).hexdigest()
    archive=ROOT/'scripts/garment_verification/garment_results_v2_backup.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        frame=pd.read_parquet(io.BytesIO(z.read('garment_results_v2/products_annotated.parquet')))
    original=pd.read_parquet(master)
    cols=[c for c in original.columns if c!='parent_asin']
    pd.testing.assert_frame_equal(original.set_index('parent_asin').sort_index(),frame.set_index('parent_asin')[cols].sort_index(),check_dtype=False)
    catalog,options=compact_frames(frame)
    for data,name in [(catalog,'shirts_catalog.parquet'),(options,'shirt_size_options.parquet')]:
        dest=ROOT/'data'/name
        tmp=dest.with_suffix('.parquet.tmp');data.to_parquet(tmp,index=False);tmp.replace(dest)
    report={'source_archive':archive.name,'original_master_preserved':True,'original_master_sha256':before,
        'original_rows':len(original),'active_products':len(catalog),'recorded_listing_options':len(options),
        'active_catalog':'data/shirts_catalog.parquet','options':'data/shirt_size_options.parquet',
        'unavailable_size_products':int(catalog.sizes.eq('').sum()),
        'note':'One record per unique shirt-candidate URL, not verified unique physical products. Uncertain/non-shirt records remain in the original master and result archive.'}
    assert hashlib.sha256(master.read_bytes()).hexdigest()==before
    (ROOT/'data/reports/compact_catalog_build.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
