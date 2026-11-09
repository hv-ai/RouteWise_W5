import unittest
from routewise.core import LABELS
from routewise.experiment import text_id, wilson_interval, choose_policy


def example(truth='Software', prediction='Software', confidence=.94):
    return {'truth':truth, 'scores':{k:confidence if k==prediction else (1-confidence)/6 for k in LABELS}}


class ExperimentTests(unittest.TestCase):
    def test_normalized_identical_tickets_have_same_id(self):
        self.assertEqual(text_id('  PRINTER\n failed  '), text_id('printer failed'))

    def test_zero_accepted_has_no_accuracy_interval(self):
        self.assertIsNone(wilson_interval(0,0))

    def test_small_perfect_sample_is_not_certainty(self):
        low, high = wilson_interval(3,3)
        self.assertLess(low,.5)
        self.assertAlmostEqual(high,1)

    def test_interval_contains_observed_fraction(self):
        low, high = wilson_interval(35,38)
        self.assertLess(low,35/38)
        self.assertGreater(high,35/38)

    def test_fallback_when_no_policy_meets_target(self):
        chosen = choose_policy([example(truth='O365') for _ in range(30)])
        self.assertTrue(chosen['fallback_review_all'])
        self.assertFalse(chosen['policy']['auto_enabled'])

    def test_small_sample_cannot_meet_minimum(self):
        self.assertTrue(choose_policy([example() for _ in range(3)])['fallback_review_all'])

    def test_eligible_policy_selected_from_calibration(self):
        chosen = choose_policy([example() for _ in range(30)])
        self.assertFalse(chosen['fallback_review_all'])
        self.assertEqual(chosen['calibration_metrics']['automated'],30)
        self.assertEqual(chosen['selection_split'],'calibration')
        self.assertFalse(chosen['target_is_guarantee'])


if __name__ == '__main__':
    unittest.main()
