"""Backfill only missing catalog sleeves from traceable local metadata text."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parent
LONG = re.compile(r'\b(?:long|full)[\s-]*sleeve(?:d|s)?\b')
SHORT = re.compile(r'\b(?:short|half)[\s-]*sleeve(?:d|s)?\b')
# A shirt whose sleeves can be raised to wear it short has long sleeves in its
# unrolled state. Do not classify generic "roll-up sleeve" text by itself.
CONVERTIBLE = re.compile(r'\bsleeves\s+can\s+be\b[^.!?]{0,60}\b(?:slipped\s+upwards|rolled\s+up)\b'
                        r'[^.!?]{0,100}\bworn\b[^.!?]{0,60}\bshort[\s-]*sleeve(?:d|s)?\b'
                        r'|\bsleeves\s+can\s+be\s+rolled\s+(?:and\s+tied\s+)?up\s+for\s+a\s+short[\s-]*sleeve\s+look\b')
NEGATIVE = re.compile(r'\b(?:not|no|without)\b[^.!?]{0,30}\b(?:long|full|short|half)[\s-]*sleeve')


def sleeve_evidence(text):
    text = re.sub(r'\s+', ' ', str(text or '').lower().replace('–','-').replace('‑','-'))
    if NEGATIVE.search(text): return 'unknown', ''
    converted = list(CONVERTIBLE.finditer(text))
    remaining = CONVERTIBLE.sub('', text)
    long_hits, short_hits = list(LONG.finditer(remaining)), list(SHORT.finditer(remaining))
    full = bool(converted or long_hits)
    half = bool(short_hits)
    if full == half: return 'unknown', ''
    matches = converted + long_hits if full else short_hits
    return ('full' if full else 'half'), '; '.join(dict.fromkeys(m.group(0) for m in matches))


def enrich_frames(master, metadata):
    for frame in (master, metadata):
        if frame.parent_asin.isna().any() or frame.parent_asin.duplicated().any():
            raise ValueError('Product IDs must be nonempty and unique')
    output = master.reset_index(drop=True).copy(deep=True)
    aligned = metadata.set_index('parent_asin').reindex(output.parent_asin)
    if 'sleeve_original' not in output: output['sleeve_original'] = output.sleeve
    if 'sleeve_source' not in output:
        output['sleeve_source'] = output.sleeve.map(lambda v: 'legacy_catalog' if v in ('full','half') else 'unresolved')
    if 'sleeve_evidence' not in output: output['sleeve_evidence'] = ''
    for index in output.index[~output.sleeve.isin(['full','half'])]:
        fields = [('title', output.at[index,'title'])]
        fields += [(name, aligned.iloc[index].get(name,'')) for name in ('features','description')]
        blob = '. '.join(str(v) for _, v in fields if pd.notna(v))
        value, evidence = sleeve_evidence(blob)
        if value != 'unknown':
            sources = [name for name, text in fields if pd.notna(text) and sleeve_evidence(str(text))[0] == value]
            output.at[index,'sleeve'] = value
            output.at[index,'sleeve_source'] = 'metadata_text'
            output.at[index,'sleeve_evidence'] = ','.join(sources) + ': ' + evidence
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='Write backed-up catalog; otherwise only report.')
    args = parser.parse_args()
    path = ROOT / 'data/shirts_master.parquet'
    metadata_path = ROOT.parent / 'data/processed/men_shirts_metadata.parquet'
    master = pd.read_parquet(path)
    metadata = pd.read_parquet(metadata_path, columns=['parent_asin','features','description'])
    metadata = metadata[metadata.parent_asin.isin(master.parent_asin)]
    enriched = enrich_frames(master, metadata)
    changed = ~enriched.sleeve.fillna('unknown').eq(master.sleeve.fillna('unknown'))
    # Verify that row identity and all non-sleeve catalog fields stay identical.
    pd.testing.assert_frame_equal(master.drop(columns=['sleeve']), enriched[master.columns].drop(columns=['sleeve']))
    report = {'products':len(master), 'changed':int(changed.sum()),
              'before':master.sleeve.value_counts(dropna=False).to_dict(),
              'after':enriched.sleeve.value_counts(dropna=False).to_dict(),
              'metadata':str(metadata_path), 'applied':args.apply,
              'policy':'Only missing sleeves filled; explicit length or convertible-long-sleeve evidence; conflicting/silent text stays unknown.'}
    qa = ROOT.parent / 'JUNK created by qoder/sleeve_repair'
    qa.mkdir(parents=True, exist_ok=True)
    enriched.loc[changed, ['parent_asin','title','sleeve_original','sleeve','sleeve_source','sleeve_evidence']].to_csv(qa / 'changed_sleeves.csv', index=False)
    if args.apply:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        backup = path.parent / 'backups' / f'shirts_master_before_sleeves_{digest[:12]}.parquet'
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists(): shutil.copy2(path, backup)
        if hashlib.sha256(backup.read_bytes()).hexdigest() != digest: raise ValueError('Backup mismatch')
        temp = path.with_suffix('.parquet.tmp')
        enriched.to_parquet(temp, index=False)
        pd.testing.assert_frame_equal(enriched, pd.read_parquet(temp))
        export_dir = path.parent / 'exports'; export_dir.mkdir(exist_ok=True)
        csv_path = export_dir / (path.stem + '.csv')
        csv_temp = csv_path.with_suffix('.csv.tmp')
        enriched.to_csv(csv_temp, index=False)
        temp.replace(path)
        csv_temp.replace(csv_path)
        report['backup'] = str(backup)
        report['original_sha256'] = digest
        report_dir = path.parent / 'reports'; report_dir.mkdir(exist_ok=True)
        (report_dir / 'sleeve_enrichment_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (qa / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
