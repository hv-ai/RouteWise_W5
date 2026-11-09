"""Offline access to recorded GPU scores. This class never runs a model."""
from copy import deepcopy
from .core import validate_scores


class RecordedRouter:
    recorded = True

    def __init__(self, tickets):
        self.records = {}
        for ticket in tickets:
            validate_scores(ticket['scores'])
            key = ticket['text'].strip()
            if key in self.records:
                raise ValueError('Duplicate recorded text; choose a specific run before replaying.')
            self.records[key] = deepcopy(ticket)

    def score(self, text):
        key = text.strip()
        if key not in self.records:
            raise ValueError('This text was not recorded. New tickets require the live GPU notebook.')
        row = self.records[key]
        return {
            'prediction':max(row['scores'],key=row['scores'].get),
            'scores':deepcopy(row['scores']), 'log_scores':deepcopy(row['log_scores']),
            'latency_ms':row['latency_ms'],
            'score_method':row['score_method'], 'calibrated':False,
            'inference_mode':'recorded_scores', 'new_inference_executed':False,
            'latency_is_historical':True,
        }
