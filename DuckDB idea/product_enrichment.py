"""Add traceable brand/size fields and product detail records from local metadata.

Recorded sizes are historical listing evidence, not live stock or image-specific
variant availability. Missing evidence never defaults to a standard size range.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_SIZES = ['XXS','XS','S','M','L','XL','XXL','3XL','4XL','5XL','6XL','7XL','8XL','One Size']
SIZE_NAMES = set(BASE_SIZES + [s+' Tall' for s in BASE_SIZES[:-1]])
TOKEN = (r'(?:one\s+size(?:\s+fits\s+all)?|extra[ -]*(?:small|large)|'
         r'[2-8]\s*x[ -]*large|[2-8]\s*xl|x{1,3}[ -]*(?:small|large)|'
         r'xxs|xs|x{1,3}l|small|medium|large|s|m|l)(?:\s+tall)?')
SIZE_RE = re.compile(r'(?<![\w])'+TOKEN+r'(?![\w])', re.I)
SIZE_SEQUENCE = re.compile(r'(?<![\w])'+TOKEN+r'(?:(?:\s*[,/;|~\-]\s*|\s+(?:to|through|and)\s+)'+TOKEN+r')+(?![\w])', re.I)
MODEL = re.compile(r'\bmodel\b.{0,100}\b(?:wears?|wearing|size)\b', re.I)
SIZE_CONTEXT = re.compile(r'\b(?:sizes?|sizing|available\s+in|offered\s+in)\s*[:=\-]?\s*$',re.I)
MEASUREMENT_LABEL = re.compile(r'^\s*(?:[:=]\s*\(?|\()\s*(?:US\b|UK\b|EU\b|\d|chest\b|bust\b|shoulder\b|length\b|neck\b)',re.I)


def clean_text(value):
    return html.unescape(re.sub(r'<[^>]*>', ' ', str(value or ''))).strip()


def canonical_size(value):
    text = str(value).strip().lower()
    tall = bool(re.search(r'\s+tall$', text))
    if tall: text = re.sub(r'\s+tall$', '', text)
    compact = re.sub(r'[\s-]', '', text)
    aliases = {'xxsmall':'XXS','xxs':'XXS','xsmall':'XS','extrasmall':'XS','xs':'XS',
               'small':'S','s':'S','medium':'M','m':'M','large':'L','l':'L',
               'extralarge':'XL','xlarge':'XL','xl':'XL','xxlarge':'XXL','xxl':'XXL',
               'xxxlarge':'3XL','xxxl':'3XL','onesize':'One Size','onesizefitsall':'One Size'}
    size = aliases.get(compact)
    if size is None:
        match = re.fullmatch(r'([2-8])(?:xl|xlarge)', compact)
        if match: size = 'XXL' if match[1]=='2' else match[1]+'XL'
    return size + (' Tall' if tall else '') if size else None


def json_value(value, expected):
    if isinstance(value,expected): return value
    try: result = json.loads(value or '')
    except (ValueError,TypeError): return expected()
    return result if isinstance(result,expected) else expected()


def extract_sizes(title, details, features, descriptions):
    found, witnesses = set(), []
    # Only product-size keys, never package dimensions or unrelated wheel sizes.
    for key, value in details.items():
        if re.sub(r'[\s_\-]','',key.lower()) not in {'size','sizename','sizemap'}: continue
        raw = clean_text(value)
        labels = [canonical_size(m.group()) for m in SIZE_RE.finditer(raw)]
        found.update(x for x in labels if x)
        witnesses.append(('details', key + ': ' + raw))
    sources = [('title',title)] + [('features',x) for x in features] + [('description',x) for x in descriptions]
    for source, value in sources:
        value = clean_text(value)
        for sentence in re.split(r'(?<=[.!?])\s+|[\r\n]+', value):
            if MODEL.search(sentence): continue
            if re.search(r'\bsize\s+small\b.*\b(?:choose|order)\b.*\bsize\s+up\b',sentence,re.I): continue
            # L/S and S/S describe sleeve length, not a size list.
            scanned = re.sub(r'\b[ls]\s*/\s*s\b',' ',sentence,flags=re.I)
            accepted = []
            for sequence in SIZE_SEQUENCE.finditer(scanned):
                accepted.extend(m.group() for m in SIZE_RE.finditer(sequence.group()))
            for match in SIZE_RE.finditer(scanned):
                before, after = scanned[:match.start()], scanned[match.end():]
                explicit = SIZE_CONTEXT.search(before[-80:])
                chart = MEASUREMENT_LABEL.match(after)
                # Titles frequently end with a single variant size. Do not read
                # ordinary "large pockets" or "medium weight" as size labels.
                trailing_title = source=='title' and not after.strip(' ,.)]') and (
                    not before or before.endswith((' ', ',', '(', '[')))
                recommendation = re.search(r'\b(?:order|choose|recommend|wears?|wearing)\b',before[-90:],re.I)
                if chart or trailing_title or (explicit and not recommendation): accepted.append(match.group())
            labels = {canonical_size(label) for label in accepted}
            labels.discard(None)
            if labels:
                found.update(labels)
                witnesses.append((source,sentence.strip()))
    ordered = [s for base in BASE_SIZES for s in (base,base+' Tall') if s in found]
    sources = list(dict.fromkeys(source for source,_ in witnesses))
    return {'sizes':'|'.join(ordered), 'size_source':','.join(sources) or 'unavailable',
            'size_evidence':json.dumps([{'source':source,'text':text} for source,text in dict.fromkeys(witnesses)],ensure_ascii=False),
            'size_note':'Sizes recorded or mentioned in the 2023 listing; check the original size chart for US/Asian sizing. Stock and variant availability are unverified.' if witnesses else 'Size information unavailable in the source listing.'}


def enrich_frames(master, audit, brands):
    for frame in (master,audit,brands):
        if frame.parent_asin.isna().any() or frame.parent_asin.duplicated().any():
            raise ValueError('Product IDs must be nonempty and unique')
    if set(master.parent_asin) != set(audit.parent_asin):
        raise ValueError('Source details must cover exactly the master product IDs')
    output = master.reset_index(drop=True).copy(deep=True)
    aligned = audit.set_index('parent_asin').loc[output.parent_asin].reset_index()
    brand_lookup = brands.set_index('parent_asin').brand.to_dict()
    size_rows, brand_rows, detail_rows = [], [], []
    for index, row in enumerate(aligned.to_dict('records')):
        details = json_value(row.get('source_details'),dict)
        features = [clean_text(v) for v in json_value(row.get('source_features'),list) if isinstance(v,str)]
        descriptions = [clean_text(v) for v in json_value(row.get('source_description'),list) if isinstance(v,str)]
        size_rows.append(extract_sizes(output.at[index,'title'],details,features,descriptions))
        brand = next((clean_text(v) for k,v in details.items() if k.strip().lower() in ('brand','brand name') and isinstance(v,str) and v.strip()),None)
        source = 'details' if brand else 'metadata_brand'
        if not brand:
            candidate = brand_lookup.get(row['parent_asin'])
            brand = clean_text(candidate) if pd.notna(candidate) and candidate else None
        if not brand:
            source = 'store'
            candidate = output.at[index,'store']
            brand = clean_text(candidate) if pd.notna(candidate) else ''
        brand_rows.append({'brand':brand,'brand_source':source if brand else 'unavailable'})
        urls = json_value(row.get('source_image_urls'),list)
        detail_rows.append({'parent_asin':row['parent_asin'], 'features_json':json.dumps(features,ensure_ascii=False),
            'description_json':json.dumps(descriptions,ensure_ascii=False),
            'details_json':json.dumps(details,ensure_ascii=False),
            'image_urls_json':json.dumps([u for u in urls if isinstance(u,str) and u.startswith(('https://','http://'))]),
            'listing_title':row.get('source_title') or '', 'metadata_year':2023})
    for frame in (pd.DataFrame(size_rows),pd.DataFrame(brand_rows)):
        for name in frame: output[name] = frame[name]
    return output,pd.DataFrame(detail_rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    master_path=ROOT/'data/shirts_master.parquet'
    source=ROOT.parent/'data/source_color_audit/metadata_color_evidence.parquet'
    brand_source=ROOT.parent/'data/processed/men_shirts_metadata.parquet'
    master=pd.read_parquet(master_path)
    output,details=enrich_frames(master,pd.read_parquet(source),pd.read_parquet(brand_source,columns=['parent_asin','brand']))
    original=[c for c in master if c not in {'sizes','size_source','size_evidence','size_note','brand','brand_source'}]
    pd.testing.assert_frame_equal(master[original],output[original])
    report={'products':len(output), 'size_labels_recorded':int(output.sizes.ne('').sum()),
            'size_raw_evidence_only':int((output.sizes.eq('') & output.size_source.ne('unavailable')).sum()),
            'size_unavailable':int(output.size_source.eq('unavailable').sum()),
            'brand_sources':output.brand_source.value_counts().to_dict(),
            'source':str(source), 'applied':args.apply,
            'note':'Size mentions and size charts are historical metadata evidence, not live variant availability.'}
    qa=ROOT.parent/'JUNK created by qoder/product_details_qa';qa.mkdir(parents=True,exist_ok=True)
    output[['parent_asin','title','brand','brand_source','sizes','size_source','size_evidence']].to_csv(qa/'size_brand_audit.csv',index=False)
    if args.apply:
        digest=hashlib.sha256(master_path.read_bytes()).hexdigest()
        backup=master_path.parent/'backups'/f'shirts_master_before_details_{digest[:12]}.parquet'
        if not backup.exists(): shutil.copy2(master_path,backup)
        if hashlib.sha256(backup.read_bytes()).hexdigest()!=digest: raise ValueError('Backup mismatch')
        details_path=master_path.parent/'product_details.parquet'
        temp=details_path.with_suffix('.parquet.tmp');details.to_parquet(temp,index=False)
        pd.testing.assert_frame_equal(details,pd.read_parquet(temp));temp.replace(details_path)
        temp=master_path.with_suffix('.parquet.tmp');output.to_parquet(temp,index=False)
        pd.testing.assert_frame_equal(output,pd.read_parquet(temp))
        export_dir=master_path.parent/'exports';export_dir.mkdir(exist_ok=True)
        csv_path=export_dir/(master_path.stem+'.csv');csv_temp=csv_path.with_suffix('.csv.tmp')
        output.to_csv(csv_temp,index=False);temp.replace(master_path);csv_temp.replace(csv_path)
        report['backup']=str(backup)
        report_dir=ROOT/'data/reports';report_dir.mkdir(exist_ok=True)
        (report_dir/'product_enrichment_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (qa/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
