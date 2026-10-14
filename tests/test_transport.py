"""Unit-sensitive comparisons and frozen event identity contracts."""
from pathlib import Path
import tempfile, unittest
from unittest.mock import patch
import numpy as np
from src import transport


class TransportChecks(unittest.TestCase):
    def run_modified(self, filename, change):
        original = transport.rd
        def read(path):
            frame = original(path)
            if Path(path).name == filename:
                frame = frame.copy()
                change(frame)
            return frame
        with tempfile.TemporaryDirectory() as out, patch.object(transport, 'rd', side_effect=read):
            transport.t1(out)

    def test_humidity_error_rejected_for_estimate_and_interval(self):
        for column in ['difference', 'low', 'high']:
            with self.subTest(column=column), self.assertRaises(AssertionError):
                def change(frame):
                    row = frame.index[frame.units.eq('g kg-1')][0]
                    frame.loc[row, column] += .05
                self.run_modified('T3_summary.csv', change)

    def test_humidity_roundoff_accepted(self):
        def change(frame):
            row = frame.index[frame.units.eq('g kg-1')][0]
            frame.loc[row, 'difference'] += 1e-12
        self.run_modified('T3_summary.csv', change)

    def test_units_must_match(self):
        def change(frame):
            row = frame.index[frame.units.eq('g kg-1')][0]
            frame.loc[row, 'units'] = 'kg'
        with self.assertRaises(AssertionError):
            self.run_modified('T3_summary.csv', change)

    def test_sample_size_must_match(self):
        with self.assertRaises(AssertionError):
            self.run_modified('T3_summary.csv', lambda f: f.__setitem__('n', f.n + 1))

    def test_event_identity_must_match(self):
        def change(frame):
            frame.loc[0, 'sid'] = 'INVALID_SID'
        with self.assertRaises(AssertionError):
            self.run_modified('T1_event_UTC_columns.csv', change)

    def test_event_utc_must_match_lag(self):
        def change(frame):
            frame.loc[0, 'UTC'] = frame.loc[1, 'UTC']
        with self.assertRaises(AssertionError):
            self.run_modified('T1_event_UTC_columns.csv', change)

    def test_shapes_must_match_without_broadcasting(self):
        with self.assertRaises(AssertionError):
            transport.eq(1., np.array([1., 1.]))

    def test_nan_support_must_match(self):
        with self.assertRaises(AssertionError):
            transport.eq([np.nan, 1.], [1., np.nan])

    def test_stored_pair_difference_must_match(self):
        def change(frame):
            row = frame.index[frame.units.eq('g kg-1')][0]
            frame.loc[row, 'difference'] += .05
        with self.assertRaises(AssertionError):
            self.run_modified('T3_common_support_event_values.csv', change)

    def test_negative_draw_indices_rejected(self):
        original = np.load
        for filename in ['T3_joint_bootstrap.npz', 'T1_bootstrap_post_side_net.npz']:
            def load(path, *args, **kwargs):
                bank = original(path, *args, **kwargs)
                if Path(path).name == filename:
                    copied = {k: bank[k].copy() for k in bank.files}
                    bank.close()
                    copied['indices'][copied['indices'] == len(copied['sid']) - 1] = -1
                    return copied
                return bank
            with self.subTest(bank=filename), tempfile.TemporaryDirectory() as out:
                with patch.object(transport.np, 'load', side_effect=load), self.assertRaises(AssertionError):
                    transport.t1(out)

    def test_kg_roundoff_keeps_its_original_scale(self):
        transport.eq([1e12, np.nan], [1e12 + .01, np.nan])
        with self.assertRaises(AssertionError):
            transport.eq([1e12], [1e12 + 1e5])


if __name__ == '__main__':
    unittest.main()
