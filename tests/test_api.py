import os
from unittest import TestCase
from unittest.mock import Mock, patch

import requests
import main
from test_recommender import entry


class ApiTests(TestCase):
    def setUp(self):
        self.client = main.app.test_client()
        main._cache.clear()

    def test_username_validation(self):
        for username in ('', 'a', '../../etc', 'name&offset=1', 'a' * 17):
            with self.subTest(username=username):
                response = self.client.get('/api/recommendations', query_string={'username': username})
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.json['error'])

    def test_missing_credentials_return_json_not_exit(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(main, 'TOKEN_FILE', '/missing/token.json'):
            response = self.client.get('/api/recommendations?username=SyntaxOverlord')
            self.assertEqual(response.status_code, 503)
            self.assertIn('MAL_CLIENT_ID', response.json['message'])
            self.assertTrue(self.client.get('/api/health').json['ok'])

    def test_client_id_auth(self):
        with patch.dict(os.environ, {'MAL_CLIENT_ID': 'example-client', 'MAL_TOKEN': 'unused'}):
            self.assertEqual(main.mal_headers(), {'X-MAL-CLIENT-ID': 'example-client'})

    def test_pagination_includes_every_status(self):
        pages = [{'data': [{'node': {'id': 1}, 'list_status': {'score': 10, 'status': 'completed'}}],
                  'paging': {'next': 'https://untrusted.invalid/not-followed'}},
                 {'data': [{'node': {'id': 2}, 'list_status': {'status': 'plan_to_watch'}}]}]
        with patch.object(main, 'mal_get', side_effect=pages) as get:
            entries = main.pull_animelist('SyntaxOverlord')
            self.assertEqual([e['id'] for e in entries], [1, 2])
            self.assertEqual(get.call_args_list[1].args[1]['offset'], 1000)

    def test_upstream_errors_are_safe_json(self):
        for upstream, expected in ((401, 503), (403, 403), (404, 404), (429, 503), (500, 502)):
            with self.subTest(upstream=upstream), patch.object(main, 'mal_headers', return_value={}), \
                    patch.object(main.requests, 'get', return_value=Mock(status_code=upstream, text='SECRET RAW BODY')), \
                    patch.object(main.time, 'sleep'):
                response = self.client.get('/api/recommendations?username=SyntaxOverlord')
                self.assertEqual(response.status_code, expected)
                self.assertNotIn('SECRET', response.get_data(as_text=True))

    def test_network_timeout(self):
        with patch.object(main, 'mal_headers', return_value={}), \
                patch.object(main.requests, 'get', side_effect=requests.Timeout), patch.object(main.time, 'sleep'):
            self.assertEqual(self.client.get('/api/recommendations?username=tester').status_code, 502)

    def test_bad_upstream_json(self):
        upstream = Mock(status_code=200)
        upstream.json.side_effect = ValueError
        with patch.object(main, 'mal_headers', return_value={}), \
                patch.object(main.requests, 'get', return_value=upstream), patch.object(main.time, 'sleep'):
            self.assertEqual(self.client.get('/api/recommendations?username=tester').status_code, 502)

    def test_empty_list(self):
        with patch.object(main, 'pull_animelist', return_value=[]):
            self.assertEqual(self.client.get('/api/recommendations?username=tester').status_code, 422)

    def test_no_positive_evidence(self):
        with patch.object(main, 'pull_animelist', return_value=[entry(1, 1)]), \
                patch.object(main, 'collaborative', return_value=[]):
            response = self.client.get('/api/recommendations?username=tester')
            self.assertEqual(response.status_code, 422)
            self.assertIn('7/10', response.json['message'])

    def test_complete_community_flow_and_prequel_filter(self):
        entries = [entry(1, 10), entry(99, 0, 'plan_to_watch')]
        details = {1: {'recommendations': [
            {'node': {'id': aid}, 'num_recommendations': 1} for aid in (99, 2, 3, 4)]},
            2: {'id': 2, 'title': 'Needs prequel', 'related_anime': [
                {'relation_type': 'prequel', 'node': {'id': 80}}]},
            3: {'id': 3, 'title': 'Good pick', 'mean': 8.2},
            4: {'id': 4, 'title': 'Explicit', 'rating': 'rx'}}
        with patch.object(main, 'pull_animelist', return_value=entries), \
                patch.object(main, 'collaborative', return_value=[]), \
                patch.object(main, 'mal_one', side_effect=lambda aid: details[aid]):
            response = self.client.get('/api/recommendations?username=SyntaxOverlord')
            self.assertEqual(response.status_code, 200)
            data = response.json
            self.assertEqual(data['method'], 'community')
            self.assertEqual([r['anime_id'] for r in data['recommendations']], [3])
            self.assertIsNone(data['recommendations'][0]['score'])
            self.assertEqual(data['profile']['rated'], 1)

    def test_metadata_cached(self):
        with patch.object(main, 'mal_get', return_value={'id': 1}) as fetch:
            self.assertEqual(main.mal_one(1), main.mal_one(1))
            fetch.assert_called_once()

    def test_health_does_not_expose_server_paths(self):
        response = self.client.get('/api/health')
        self.assertNotIn('db', response.json)
        self.assertIn('no-store', response.headers['Cache-Control'])

    def test_pages_and_assets(self):
        for path in ('/', '/results.html', '/styles.css', '/app.js'):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                response.close()

    def test_deadline_stops_upstream_work(self):
        with main.app.test_request_context('/api/recommendations?username=tester'), \
                patch.object(main, 'mal_headers', return_value={}), \
                patch.object(main.time, 'sleep'), patch.object(main.requests, 'get') as fetch:
            main.g.deadline = 0
            with self.assertRaises(main.APIError) as raised:
                main.mal_get('anime/1')
            self.assertEqual(raised.exception.status, 504)
            fetch.assert_not_called()

    def test_partial_results_survive_deadline(self):
        candidates = [{'anime_id': aid, 'score': 8, 'source': 'similar_viewers',
                       'reason': 'Three similar viewers rated this anime.'} for aid in (2, 3)]
        with patch.object(main, 'pull_animelist', return_value=[entry(1, 10)]), \
                patch.object(main, 'collaborative', return_value=candidates), \
                patch.object(main, 'mal_one', side_effect=[{'id': 2, 'title': 'Match'},
                                                         main.APIError('Timeout', 504)]):
            response = self.client.get('/api/recommendations?username=tester')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json['recommendations']), 1)
            self.assertIn('too long', response.json['warnings'][0])
