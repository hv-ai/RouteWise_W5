import unittest
from datetime import datetime, timedelta, timezone
from routewise.core import (
    LABELS, Policy, Desk, normalize_scores, policy_metrics,
    suspected_incidents, similarity, ticket_terms,
)


def scores(top="Software", value=.94):
    return {label: value if label == top else (1-value)/6 for label in LABELS}


class RoutingTests(unittest.TestCase):
    def test_extreme_log_scores_remain_finite(self):
        result = normalize_scores({k: -10000-i for i, k in enumerate(LABELS)})
        self.assertAlmostEqual(sum(result.values()), 1)
        self.assertEqual(max(result, key=result.get), LABELS[0])

    def test_missing_labels_cannot_be_routed(self):
        with self.assertRaises(ValueError):
            Policy().decide({"Software": 1})

    def test_nan_cannot_be_routed(self):
        bad = scores(); bad["Software"] = float('nan')
        with self.assertRaises(ValueError):
            Policy().decide(bad)

    def test_review_is_default_even_with_high_score(self):
        self.assertEqual(Policy().decide(scores())["status"], "needs_review")

    def test_explicit_policy_accepts_clear_case(self):
        self.assertEqual(Policy(auto_enabled=True).decide(scores())["status"], "automatic")

    def test_close_candidates_require_review(self):
        near = {k: .02 for k in LABELS}
        near["Software"] = .46; near["O365"] = .44
        self.assertEqual(Policy(.4, .1, True).decide(near)["status"], "needs_review")

    def test_low_top_score_requires_review(self):
        self.assertEqual(Policy(auto_enabled=True).decide(scores(value=.4))["status"], "needs_review")

    def test_negative_threshold_rejected(self):
        with self.assertRaises(ValueError):
            Policy(-.1)

    def test_no_accepted_tickets_does_not_mean_perfect_accuracy(self):
        result = policy_metrics([{"truth": "Software", "scores": scores()}], Policy())
        self.assertEqual(result["coverage"], 0)
        self.assertIsNone(result["automatic_accuracy"])

    def test_coverage_and_errors_account_for_every_ticket(self):
        rows = [{"truth":"O365", "scores":scores()},
                {"truth":"Software", "scores":scores(value=.4)}]
        result = policy_metrics(rows, Policy(auto_enabled=True))
        self.assertEqual(result["automated"], 1)
        self.assertEqual(result["review"], 1)
        self.assertEqual(result["automatic_errors"], 1)
        self.assertEqual(result["automatic_accuracy"], 0)

    def test_human_correction_keeps_original_prediction(self):
        desk = Desk(); record = desk.add("Install Acrobat", scores(), Policy())
        desk.review(record['id'], "O365")
        self.assertEqual(record['predicted_queue'], "Software")
        self.assertEqual(record['human_queue'], "O365")
        self.assertEqual(record['status'], "reviewed")
        self.assertEqual(len(desk.reviews), 1)

    def test_review_audit_preserves_previous_decision(self):
        desk = Desk(); record = desk.add("Install Acrobat", scores(), Policy())
        desk.review(record['id'], "Software"); desk.review(record['id'], "O365")
        self.assertEqual(desk.reviews[-1]['previous_human_queue'], "Software")


class IncidentTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 7, 17, tzinfo=timezone.utc)

    def ticket(self, index, text="Outlook email sending failed for our team this morning.", offset=0):
        return {'id':str(index), 'text':text,
                'timestamp':(self.now+timedelta(minutes=offset)).isoformat()}

    def test_three_similar_reports_form_suspected_group(self):
        rows = [self.ticket(i) for i in range(3)]
        found = suspected_incidents(rows, now=self.now)
        self.assertEqual(found[0]['count'], 3)
        self.assertFalse(found[0]['confirmed'])

    def test_repeated_id_does_not_inflate_incident(self):
        self.assertEqual(suspected_incidents([self.ticket(1)]*3, now=self.now), [])

    def test_old_and_future_reports_excluded(self):
        rows = [self.ticket(1), self.ticket(2, offset=-11), self.ticket(3, offset=1)]
        self.assertEqual(suspected_incidents(rows, now=self.now), [])

    def test_unrelated_requests_do_not_form_group(self):
        rows = [self.ticket(1), self.ticket(2, "Printer paper jam requires repair"),
                self.ticket(3, "Retire old server remove monitoring")]
        self.assertEqual(suspected_incidents(rows, now=self.now), [])

    def test_stopword_only_text_has_zero_similarity(self):
        self.assertEqual(similarity(ticket_terms('the and my'), ticket_terms('the and my')), 0)

    def test_empty_stream_has_no_incidents(self):
        self.assertEqual(suspected_incidents([]), [])


if __name__ == '__main__':
    unittest.main()
