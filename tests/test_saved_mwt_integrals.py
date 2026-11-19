"""Integrity contracts for the published single-event rate-integral example."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'paper_2026/physics/vertical_extension/scripts'))
from verify_saved_example import EXAMPLE_EVENTS, RATE_METRICS, saved_integral_errors


class SavedIntegralTests(unittest.TestCase):
    def setUp(self):
        sid, event = next(iter(EXAMPLE_EVENTS))
        self.requests = pd.DataFrame(dict(metric=m, sid=sid, UTC_s=0, value=1.)
                                     for m in RATE_METRICS)
        self.reference = pd.DataFrame(dict(metric=m, sid=sid, event_id=event, value=1.)
                                      for m in RATE_METRICS)
        self.derived = self.reference.iloc[:1].drop(columns='metric').assign(valid=True)

    def check_example(self):
        return saved_integral_errors(self.requests, self.reference, lambda series: self.derived)

    def test_complete_keys(self):
        self.assertEqual(self.check_example(), dict.fromkeys(RATE_METRICS, 0.))

    def test_no_matching_event(self):
        self.derived.loc[:, 'event_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'wrong derived event keys'):
            self.check_example()

    def test_empty_derived(self):
        self.derived.loc[:, 'valid'] = False
        with self.assertRaisesRegex(ValueError, 'empty or duplicate derived'):
            self.check_example()

    def test_missing_metric(self):
        self.requests = self.requests.iloc[1:]
        with self.assertRaisesRegex(ValueError, 'missing request metrics'):
            self.check_example()

    def test_duplicate_saved_key(self):
        self.reference = pd.concat([self.reference, self.reference.iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'duplicate saved event keys'):
            self.check_example()

    def test_nonfinite(self):
        self.reference.loc[0, 'value'] = np.nan
        self.derived.loc[:, 'value'] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            self.check_example()
