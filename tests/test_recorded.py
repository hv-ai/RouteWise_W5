import unittest
from routewise.core import LABELS
from routewise.recorded import RecordedRouter


class RecordedTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            'text':'Install Acrobat',
            'scores':{k:.94 if k=='Software' else .01 for k in LABELS},
            'log_scores':{k:-float(i+1) for i,k in enumerate(LABELS)},
            'latency_ms':500, 'score_method':'test fixture',
        }

    def test_recorded_result_cannot_claim_new_inference(self):
        result=RecordedRouter([self.row]).score('Install Acrobat')
        self.assertFalse(result['new_inference_executed'])
        self.assertTrue(result['latency_is_historical'])

    def test_unknown_text_is_not_given_invented_scores(self):
        with self.assertRaises(ValueError):
            RecordedRouter([self.row]).score('A different ticket')

    def test_result_mutation_does_not_change_recorded_scores(self):
        replay=RecordedRouter([self.row]); replay.score('Install Acrobat')['scores']['Software']=0
        self.assertEqual(replay.score('Install Acrobat')['scores']['Software'],.94)

    def test_duplicate_records_are_rejected(self):
        with self.assertRaises(ValueError): RecordedRouter([self.row,self.row])
