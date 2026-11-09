"""Reproducible data preparation and calibration-only policy selection."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from .core import LABELS, Policy, policy_metrics


def text_id(text):
    normalized = re.sub(r'\s+', ' ', text.strip().casefold())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def recover_dataset(backup_dir):
    """Reconstruct the dataset from the original 468 train + 117 validation rows."""
    root = Path(backup_dir)
    records = json.loads((root/'TRAIN.json').read_text())
    rows = []
    for record in records:
        users = [m['content'] for m in record['messages'] if m['role'] == 'user']
        labels = [m['content'] for m in record['messages'] if m['role'] == 'assistant']
        if len(users) != 1 or len(labels) != 1 or not users[0].startswith('Support Ticket: '):
            raise ValueError('Unexpected original training-record format.')
        rows.append({'text':users[0].removeprefix('Support Ticket: '), 'label':labels[0]})
    with (root/'val_split.csv').open(newline='') as file:
        rows.extend({'text':r['text'], 'label':r['label']} for r in csv.DictReader(file))
    seen = set()
    for row in rows:
        if row['label'] not in LABELS or not row['text'].strip():
            raise ValueError('Missing text or invalid routing label.')
        row['id'] = text_id(row['text'])
        if row['id'] in seen:
            raise ValueError('Duplicate normalized ticket text: group duplicates before splitting.')
        seen.add(row['id'])
    return sorted(rows, key=lambda row: row['id'])


def split_dataset(rows, seed=20261108):
    from sklearn.model_selection import train_test_split
    training, remainder = train_test_split(
        rows, test_size=.4, stratify=[r['label'] for r in rows], random_state=seed)
    calibration, test = train_test_split(
        remainder, test_size=.5, stratify=[r['label'] for r in remainder], random_state=seed)
    partitions = {'train':training, 'calibration':calibration, 'test':test}
    ids = {name: {r['id'] for r in split} for name, split in partitions.items()}
    assert not (ids['train'] & ids['calibration'] or ids['train'] & ids['test']
                or ids['calibration'] & ids['test'])
    for split in partitions.values():
        assert {r['label'] for r in split} == set(LABELS)
    assert sum(map(len, partitions.values())) == len(rows)
    return partitions


def wilson_interval(correct, total, z=1.959963984540054):
    if total < 0 or correct < 0 or correct > total:
        raise ValueError('Invalid success counts.')
    if total == 0:
        return None
    p = correct/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0, center-radius), min(1, center+radius)]


def choose_policy(calibration_rows, target_accuracy=.90, min_automated=20):
    """A fixed finite grid. Uses calibration rows only, with a review-all fallback.

    The target is an empirical selection rule, NOT an accuracy guarantee.
    """
    if not .5 <= target_accuracy <= 1 or min_automated < 1:
        raise ValueError('Invalid selection target.')
    if not calibration_rows:
        raise ValueError('Calibration examples are required.')
    candidates = []
    for threshold in [.40, .50, .60, .70, .80, .85, .90, .95, .99]:
        for margin in [0, .10, .20, .30]:
            policy = Policy(threshold, margin, True)
            metrics = policy_metrics(calibration_rows, policy)
            candidates.append({'threshold':threshold, 'min_margin':margin,
                               'auto_enabled':True, **metrics})
    eligible = [c for c in candidates if c['automated'] >= min_automated
                and c['automatic_accuracy'] >= target_accuracy]
    if eligible:
        best = max(eligible, key=lambda c:(c['automated'], -c['automatic_errors'],
                                          c['threshold'], c['min_margin']))
        chosen = {k:best[k] for k in ['threshold','min_margin','auto_enabled']}
        calibration_metrics = policy_metrics(calibration_rows, Policy(**chosen))
    else:
        chosen = {'threshold':1.0, 'min_margin':1.0, 'auto_enabled':False}
        calibration_metrics = policy_metrics(calibration_rows, Policy(**chosen))
    return {
        'policy':chosen, 'selection_split':'calibration', 'target_accuracy':target_accuracy,
        'min_automated':min_automated, 'target_is_guarantee':False,
        'fallback_review_all':not bool(eligible),
        'calibration_metrics':calibration_metrics, 'grid':candidates,
    }
