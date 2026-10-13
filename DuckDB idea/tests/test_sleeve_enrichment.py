import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enrich_sleeves import sleeve_evidence, enrich_frames


class SleeveEvidenceTests(unittest.TestCase):
    def test_rollable_long_sleeve_is_not_mistaken_for_short(self):
        value, evidence = sleeve_evidence('The sleeves can be easily slipped upwards and thus worn the shirt as a short-sleeved shirt.')
        self.assertEqual(value, 'full')
        self.assertIn('slipped upwards', evidence)

    def test_explicit_lengths_and_conflicting_or_missing_evidence(self):
        for text, expected in [('Long-sleeved cotton shirt', 'full'),
                               ('Full sleeves', 'full'), ('Short sleeve linen shirt', 'half'),
                               ('Half-sleeved shirt', 'half'),
                               ('Choose long sleeve or short sleeve', 'unknown'),
                               ('Longsleeve/Shortsleeve Shirt', 'unknown'),
                               ('Longsleeve Shirt', 'full'),
                               ('The sleeves can be rolled and tied up for a short sleeve look', 'full'),
                               ('Sleeveless shirt', 'unknown'), ('Classic shirt', 'unknown'),
                               ('Not a long sleeve shirt', 'unknown')]:
            with self.subTest(text=text):
                self.assertEqual(sleeve_evidence(text)[0], expected)

    def test_enrichment_is_id_aligned_conservative_and_preserves_other_fields(self):
        master = pd.DataFrame({'parent_asin':['A','B','C','D'], 'sleeve':['unknown','half','unknown','unknown'],
                               'color':['pink','blue','white','red'], 'title':['Shirt']*4})
        metadata = pd.DataFrame({'parent_asin':['D','B','A'], 'features':['Long sleeve and short sleeve options',
                                    'Long sleeve shirt','Full-sleeved shirt'], 'description':['']*3})
        out = enrich_frames(master, metadata)
        self.assertEqual(out.sleeve.tolist(), ['full','half','unknown','unknown'])
        self.assertEqual(out.sleeve_original.tolist(), master.sleeve.tolist())
        self.assertEqual(out.color.tolist(), master.color.tolist())
        self.assertEqual(out.loc[0,'sleeve_source'], 'metadata_text')
        self.assertEqual(out.loc[2,'sleeve_source'], 'unresolved')
        pd.testing.assert_frame_equal(enrich_frames(out, metadata), out)

    def test_duplicate_metadata_ids_are_rejected(self):
        master = pd.DataFrame({'parent_asin':['A'], 'sleeve':['unknown'], 'title':['Shirt']})
        metadata = pd.DataFrame({'parent_asin':['A','A'], 'features':['Full sleeve','Half sleeve']})
        with self.assertRaises(ValueError): enrich_frames(master, metadata)


if __name__ == '__main__': unittest.main()
