import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path

from recommender import collaborative, community_candidates, rated_entries, seeds, _load_model


def entry(aid, score=0, status='completed', title=None):
    return {'id': aid, 'score': score, 'status': status, 'title': title or f'Anime {aid}'}


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'ratings.db'
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('CREATE TABLE animelists (username TEXT, anime_id INTEGER, score REAL)')
            # Neighbors agree with our 10, 8, 2 profile. Candidate 4 has three
            # supporters; candidate 5 has just one overenthusiastic supporter.
            for user in ('a', 'b', 'c'):
                conn.executemany('INSERT INTO animelists VALUES (?, ?, ?)',
                                 [(user, aid, score) for aid, score in ((1, 10), (2, 8), (3, 2), (4, 9))])
            conn.execute('INSERT INTO animelists VALUES ("a", 5, 10)')
            conn.executemany('INSERT INTO animelists VALUES (?, ?, ?)',
                             [('ME', aid, score) for aid, score in ((1, 10), (2, 8), (3, 2), (6, 10))])
        self.entries = [entry(1, 10), entry(2, 8), entry(3, 2)]

    def tearDown(self):
        _load_model.cache_clear()
        self.temp.cleanup()

    def test_only_real_ratings_and_no_watchlist_taste(self):
        entries = [entry(1), entry(2, 9, 'plan_to_watch'), entry(3, 2, 'dropped')]
        self.assertEqual(rated_entries(entries), {3: 2})

    def test_confidence_and_self_exclusion(self):
        results = collaborative(self.entries, 'me', self.db)
        self.assertEqual([r['anime_id'] for r in results], [4])
        self.assertEqual(results[0]['support'], 3)
        self.assertLess(results[0]['score'], 9)

    def test_all_list_statuses_are_excluded(self):
        for status in ('watching', 'completed', 'dropped', 'on_hold', 'plan_to_watch'):
            with self.subTest(status=status):
                self.assertEqual(collaborative(self.entries + [entry(4, 0, status)], 'me', self.db), [])

    def test_missing_database_is_not_created(self):
        path = Path(self.temp.name) / 'missing.db'
        self.assertEqual(collaborative(self.entries, 'me', path), [])
        self.assertFalse(path.exists())

    def test_constant_ratings_have_no_fake_similarity(self):
        self.assertEqual(collaborative([entry(1, 10), entry(2, 10), entry(3, 10)], 'me', self.db), [])

    def test_seeds_exclude_dislikes_and_dropped(self):
        self.assertEqual([e['id'] for e in seeds([entry(1, 10), entry(2, 1), entry(3, 10, 'dropped')])], [1])

    def test_unrated_completion_is_implicit_not_a_score(self):
        entries = [entry(1), entry(2, 0, 'plan_to_watch'), entry(3, 0, 'watching')]
        self.assertEqual([s['id'] for s in seeds(entries)], [1])
        self.assertEqual(rated_entries(entries), {})
        self.assertEqual(entries[0]['score'], 0)

    def test_community_votes_are_normalized_per_seed(self):
        def rec(aid, votes): return {'node': {'id': aid}, 'num_recommendations': votes}
        details = [(entry(1, 10), {'recommendations': [rec(10, 100), rec(11, 100)]}),
                   (entry(2, 10), {'recommendations': [rec(12, 2)]})]
        results = community_candidates(details, {11})
        self.assertEqual([r['anime_id'] for r in results], [12, 10])
        self.assertIsNone(results[0]['score'])
        self.assertIn('Anime 2', results[0]['reason'])
