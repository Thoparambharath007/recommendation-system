import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.vendor'))
from merge_vision import merge_frames
from query_engine import Catalog, validate_sql
from llm_sql import parse_plan, SQLGenerator
from request_policy import single_color_intent
from unittest.mock import patch


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.master = pd.DataFrame([
            dict(parent_asin='A', image_url='https://example.com/a', color='wrong', pattern=None,
                 color_source='text', title='Original A'),
            dict(parent_asin='B', image_url='https://example.com/b', color='red', pattern=None,
                 color_source='text', title='Original B')])
        self.results = pd.DataFrame([
            dict(parent_asin='B', image_url='https://example.com/b', color='', pattern='unknown', status='mask_failed'),
            dict(parent_asin='A', image_url='https://example.com/a', color='green|white',
                 pattern='horizontal_stripes', status='needs_review')])

    def test_merge_uses_ids_and_keeps_uncertain_predictions(self):
        out = merge_frames(self.master, self.results)
        self.assertEqual(out.parent_asin.tolist(), ['A', 'B'])
        self.assertEqual(out.color.tolist(), ['green|white', ''])
        self.assertEqual(out.pattern.tolist(), ['horizontal_stripes', 'unknown'])
        self.assertEqual(out.title.tolist(), self.master.title.tolist())
        self.assertEqual(out.color_original.tolist(), ['wrong', 'red'])
        self.assertEqual(out.vision_status.tolist(), ['needs_review', 'mask_failed'])

    def test_bad_join_is_rejected(self):
        for results in [self.results.iloc[:1], pd.concat([self.results, self.results.iloc[:1]])]:
            with self.assertRaises(ValueError): merge_frames(self.master, results)
        bad = self.results.copy(); bad.loc[0, 'image_url'] = 'https://example.com/wrong'
        with self.assertRaises(ValueError): merge_frames(self.master, bad)


class SQLTests(unittest.TestCase):
    columns = {'parent_asin', 'color', 'pattern', 'price_inr_est', 'average_rating', 'vision_status'}

    def test_exclusive_color_intent_keeps_shades_and_multi_color_requests(self):
        self.assertEqual(single_color_intent('plain black shirt uder 1500'),('black',True))
        self.assertEqual(single_color_intent('plain light blue shirt'),('light blue',True))
        self.assertEqual(single_color_intent('only dark green shirts'),('dark green',False))
        self.assertEqual(single_color_intent('plain black and white shirts'),(None,False))
        self.assertEqual(single_color_intent('plain shirt without black'),(None,False))

    def test_exact_color_tokens_are_valid(self):
        q = "SELECT parent_asin FROM shirts WHERE contains('|' || lower(coalesce(color,'')) || '|', '|maroon|') AND pattern = 'plain'"
        self.assertIn('SELECT', validate_sql(q, self.columns))

    def test_model_color_substring_is_normalized_to_exact_token(self):
        for condition in ["color LIKE '%blue%'", "lower(color) ILIKE '%BLUE%'", "contains(color,'blue')",
                          "trim(lower(color)) = 'blue'", "color IN ('blue','white')"]:
            result=validate_sql('SELECT parent_asin FROM shirts WHERE '+condition,self.columns)
            self.assertIn('|blue|',result)
            self.assertNotIn('%blue%',result)
        with self.assertRaises(ValueError):
            validate_sql("SELECT parent_asin FROM shirts WHERE color LIKE '%bl%'",self.columns)

    def test_model_pattern_alias_matches_real_vocabulary(self):
        for condition in ["lower(pattern) LIKE '%striped%'", "pattern='striped'", "pattern IN ('striped','plaid')"]:
            result=validate_sql('SELECT parent_asin FROM shirts WHERE '+condition,self.columns)
            self.assertIn('horizontal_stripes',result)
            self.assertIn('vertical_stripes',result)
        with self.assertRaises(ValueError):
            validate_sql("SELECT parent_asin FROM shirts WHERE pattern='fabricated_pattern'",self.columns)

    def test_unsafe_or_unbounded_shape_is_rejected(self):
        queries = ["DELETE FROM shirts", "SELECT parent_asin FROM shirts; DROP TABLE shirts",
                   "SELECT parent_asin FROM read_parquet('other.parquet')", "SELECT parent_asin FROM other",
                   "SELECT parent_asin FROM shirts JOIN shirts s USING(parent_asin)",
                   "SELECT parent_asin FROM shirts WHERE button_color='white'",
                   "SELECT parent_asin FROM shirts WHERE getenv('HOME') = 'x'",
                   "SELECT * FROM shirts", "SELECT parent_asin FROM shirts LIMIT 1",
                   "SELECT parent_asin FROM shirts WHERE parent_asin IN (SELECT parent_asin FROM shirts)",
                   "WITH s AS (SELECT parent_asin FROM shirts) SELECT parent_asin FROM s",
                   "SELECT parent_asin FROM shirts WHERE random() > 0.5"]
        for q in queries:
            with self.subTest(q=q), self.assertRaises(ValueError): validate_sql(q, self.columns)

    def test_search_on_parquet_ranks_without_merging_shades(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent / 'JUNK created by qoder') as d:
            p = Path(d) / 'catalog.parquet'
            pd.DataFrame([dict(parent_asin=x, title=x, image_url='https://example.com/'+x,
                color=color, pattern='plain', store='Example', sleeve='full', fabric='cotton',
                fit_type='regular fit', pocket='single', price_usd=10., price_inr_est=830.,
                price_source='exact', average_rating=rating, rating_number=n, review_count=n,
                vision_status=status, color_source='vision_v3_2', pattern_source='vision_v3_2')
                for x, color, rating, n, status in [('A','maroon|white',4.8,100,'ok'),
                     ('B','red',5.,3,'ok'),('C','maroon',4.,50,'needs_review')]]).to_parquet(p)
            catalog = Catalog(p)
            q = "SELECT parent_asin FROM shirts WHERE contains('|' || color || '|','|maroon|')"
            result = catalog.search(q, limit=10)
            self.assertEqual(result['total_matches'], 2)
            self.assertEqual([x['parent_asin'] for x in result['products']], ['A','C'])
            eq_result = catalog.search("SELECT parent_asin FROM shirts WHERE trim(lower(color)) = 'maroon'")
            self.assertEqual(eq_result['total_matches'],2)
            strict = catalog.search(q, limit=10, include_uncertain=False)
            self.assertEqual(strict['total_matches'], 1)
            popular = catalog.search('SELECT parent_asin FROM shirts', sort='most_reviewed')
            self.assertEqual([x['parent_asin'] for x in popular['products']], ['A','C','B'])
            catalog.search("SELECT parent_asin FROM shirts WHERE average_rating >= 4")


class LLMTests(unittest.TestCase):
    def test_empty_or_nontext_model_content_is_rejected(self):
        for content in [None,{},'']:
            with self.assertRaises(ValueError): parse_plan(content)

    def test_nvidia_lightning_requests_direct_json_without_thinking(self):
        class Response:
            status_code = 200
            def json(self): return {'choices':[{'message':{'content':'{"sql":"SELECT parent_asin FROM shirts","explanation":"All shirts","unsupported":[]}'}}]}
        class Session:
            def post(self,url,**kwargs): self.sent=kwargs['json']; return Response()
        session=Session()
        SQLGenerator(base_url='https://integrate.api.nvidia.com/v1',api_key='fixture',
            model='nvidia/nemotron-3.5-lightning-30b-a3b',session=session).generate('shirts',{})
        self.assertEqual(session.sent['chat_template_kwargs'],{'enable_thinking':False})

    def test_provider_keys_are_not_reused_across_endpoints(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY':'openai-only'},clear=True):
            self.assertEqual(SQLGenerator(base_url='https://integrate.api.nvidia.com/v1').api_key,'')
            self.assertEqual(SQLGenerator(base_url='https://openrouter.ai/api/v1').api_key,'')
            self.assertEqual(SQLGenerator(base_url='https://api.openai.com/v1').api_key,'openai-only')
        with patch.dict('os.environ', {'NVIDIA_API_KEY':'nvidia-only'},clear=True):
            self.assertEqual(SQLGenerator(base_url='https://integrate.api.nvidia.com/v1').api_key,'nvidia-only')

    def test_invalid_plan_and_unsupported_fields(self):
        for raw in ['not json', '{"sql": 12}', '{"sql":"SELECT parent_asin FROM shirts","unsupported":"x"}']:
            with self.assertRaises(ValueError): parse_plan(raw)
        p = parse_plan('{"sql":"","explanation":"No button color field","unsupported":["button color"]}')
        self.assertEqual(p['unsupported'], ['button color'])

    def test_openai_compatible_adapter_uses_schema_and_preserves_query(self):
        class Response:
            status_code = 200
            def json(self):
                return {'choices':[{'message':{'content':json.dumps({'sql':'SELECT parent_asin FROM shirts',
                         'explanation':'All shirts','unsupported':[]})}}]}
        class Session:
            def post(self, url, **kwargs):
                self.sent = kwargs['json']; self.url = url
                return Response()
        session = Session()
        generator = SQLGenerator(base_url='https://example.com/v1', api_key='test', model='test-model', session=session)
        result = generator.generate('Find maroon shirts', {'columns':['parent_asin','color']})
        self.assertEqual(session.sent['messages'][-1]['content'], 'Find maroon shirts')
        self.assertEqual(session.sent['response_format'], {'type':'json_object'})
        self.assertEqual(result['sql'], 'SELECT parent_asin FROM shirts')


if __name__ == '__main__': unittest.main()
