"""ID-aligned, backed-up replacement of catalog attributes from Kaggle results."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parent
MASTER = ROOT / 'data/shirts_master.parquet'


def merge_frames(master, predictions):
    required = {'parent_asin', 'image_url', 'color', 'pattern', 'status'}
    if not required <= set(predictions): raise ValueError('Vision results lack required fields')
    for frame in (master, predictions):
        if frame.parent_asin.isna().any() or frame.parent_asin.duplicated().any():
            raise ValueError('Product IDs must be nonempty and unique')
        if frame.parent_asin.astype(str).str.strip().eq('').any():
            raise ValueError('Product IDs must be nonempty')
    if set(master.parent_asin) != set(predictions.parent_asin):
        raise ValueError('Vision results must cover exactly the master product IDs')
    aligned = predictions.set_index('parent_asin').loc[master.parent_asin].reset_index()
    if not aligned.image_url.fillna('').eq(master.image_url.fillna('').reset_index(drop=True)).all():
        raise ValueError('Image URLs differ for matched product IDs')
    output = master.reset_index(drop=True).copy(deep=True)
    for col in ['color', 'pattern', 'color_source']:
        if col in output and col + '_original' not in output:
            output[col + '_original'] = output[col]
    usable = aligned.status.isin(['ok', 'needs_review'])
    output['color'] = aligned.color.fillna('').where(usable, '')
    output['pattern'] = aligned.pattern.fillna('unknown').where(usable, 'unknown')
    output['vision_status'] = aligned.status
    output['color_source'] = ['vision_v3_2_candidate' if status == 'ok' else
        'vision_v3_2_uncertain' if status == 'needs_review' else 'vision_unresolved'
        for status in aligned.status]
    output['pattern_source'] = output.color_source.where(output.pattern.ne('unknown'), 'vision_unresolved')
    for field in ['color_main', 'color_all', 'color_candidates', 'color_status',
                  'pattern_candidate', 'pattern_score', 'pattern_margin', 'mask_quality',
                  'palette_ambiguous_share', 'unmatched_share', 'omitted_share']:
        if field in aligned: output[field] = aligned[field]
    if 'reason' in aligned: output['vision_reason'] = aligned.reason.fillna('')
    return output


def merge_archive(archive, master_path=MASTER):
    archive, master_path = Path(archive), Path(master_path)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith('/shirt_attributes.parquet')]
        if len(names) != 1: raise ValueError('Backup must contain one shirt_attributes.parquet')
        predictions = pd.read_parquet(io.BytesIO(z.read(names[0])))
    master = pd.read_parquet(master_path)
    merged = merge_frames(master, predictions)
    digest = hashlib.sha256(master_path.read_bytes()).hexdigest()
    backup_dir = master_path.parent / 'backups'; backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / ('shirts_master_before_vision_' + digest[:12] + '.parquet')
    if not backup.exists(): shutil.copy2(master_path, backup)
    if hashlib.sha256(backup.read_bytes()).hexdigest() != digest:
        raise ValueError('Backup does not match pre-merge master')
    temp = master_path.with_suffix('.parquet.tmp')
    merged.to_parquet(temp, index=False)
    check = pd.read_parquet(temp)
    pd.testing.assert_frame_equal(merged, check)
    csv_path = master_path.with_suffix('.csv')
    csv_temp = csv_path.with_suffix('.csv.tmp')
    merged.to_csv(csv_temp, index=False)
    temp.replace(master_path); csv_temp.replace(csv_path)
    summary = {'products':len(merged), 'matched_ids':len(predictions),
        'nonempty_color':int(merged.color.ne('').sum()),
        'known_pattern':int(merged.pattern.ne('unknown').sum()),
        'vision_status_counts':merged.vision_status.value_counts().to_dict(),
        'backup':str(backup.resolve()), 'original_sha256':digest,
        'master_sha256':hashlib.sha256(master_path.read_bytes()).hexdigest(),
        'source_archive':str(archive.resolve()),
        'policy':'Include needs_review predictions with provenance; failed masks stay unknown.'}
    (master_path.parent / 'vision_merge_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_archive(args.archive), indent=2))
