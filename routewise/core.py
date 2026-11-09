"""Pure Python routing policy, review records, and lexical incident grouping.

No model calls, network access, or simulated model scores live in this module.
"""
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Mapping

LABELS = (
    "Support general", "Fileservice", "O365", "EOL", "Software",
    "Active Directory", "Computer-Services",
)
SYSTEM_PROMPT = (
    "You are an IT helpdesk ticket routing assistant. "
    "Given a support ticket, respond with exactly one of the following categories: "
    "Support general, Fileservice, O365, EOL, Software, Active Directory, Computer-Services."
)


def normalize_scores(log_scores: Mapping[str, float], temperature: float = 1.0) -> dict:
    if set(log_scores) != set(LABELS):
        raise ValueError("Scores must contain each of the seven labels exactly once.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    values = {k: float(v) for k, v in log_scores.items()}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite model score; send this ticket for review.")
    top = max(values.values())
    weights = {k: math.exp((v - top) / temperature) for k, v in values.items()}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def validate_scores(scores: Mapping[str, float]) -> None:
    if set(scores) != set(LABELS):
        raise ValueError("Missing or unknown routing label.")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
        raise ValueError("Invalid routing score.")
    if not math.isclose(sum(scores.values()), 1, abs_tol=1e-5):
        raise ValueError("Routing scores must sum to one.")


@dataclass(frozen=True)
class Policy:
    threshold: float = 0.80
    min_margin: float = 0.20
    auto_enabled: bool = False

    def __post_init__(self):
        if not all(math.isfinite(v) and 0 <= v <= 1
                   for v in (self.threshold, self.min_margin)):
            raise ValueError("Threshold and margin must be between zero and one.")

    def decide(self, scores: Mapping[str, float]) -> dict:
        validate_scores(scores)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        label, score = ranked[0]
        margin = score - ranked[1][1]
        automatic = self.auto_enabled and score >= self.threshold and margin >= self.min_margin
        if not self.auto_enabled:
            reason = "Automatic routing is disabled until you enable the experimental policy."
        elif score < self.threshold:
            reason = "Top candidate score is below the review threshold."
        elif margin < self.min_margin:
            reason = "The two leading queues are too close."
        else:
            reason = "Meets the experimental score and margin thresholds."
        return {
            "predicted_queue": label, "top_score": score, "margin": margin,
            "status": "automatic" if automatic else "needs_review", "reason": reason,
            "policy": {"threshold": self.threshold, "min_margin": self.min_margin,
                       "auto_enabled": self.auto_enabled, "validated": False},
        }


def policy_metrics(records: list[dict], policy: Policy) -> dict:
    if not records:
        raise ValueError("No scored examples supplied.")
    accepted = []
    for row in records:
        if row["truth"] not in LABELS:
            raise ValueError("Evaluation truth must be a routing label.")
        decision = policy.decide(row["scores"])
        if decision["status"] == "automatic":
            accepted.append(decision["predicted_queue"] == row["truth"])
    n = len(accepted)
    return {
        "total": len(records), "automated": n, "review": len(records) - n,
        "coverage": n / len(records),
        "automatic_accuracy": sum(accepted) / n if n else None,
        "automatic_errors": n - sum(accepted),
    }


@dataclass
class Desk:
    tickets: list = field(default_factory=list)
    reviews: list = field(default_factory=list)

    def add(self, text: str, scores: dict, policy: Policy, timestamp=None, source="manual"):
        if not text.strip():
            raise ValueError("Enter a ticket first.")
        timestamp = timestamp or datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            raise ValueError("Ticket timestamps need a timezone.")
        decision = policy.decide(scores)
        record = {
            "id": f"T{len(self.tickets) + 1:04d}", "text": text,
            "timestamp": timestamp.isoformat(), "source": source,
            "scores": dict(scores), **decision,
            "human_queue": None,
        }
        self.tickets.append(record)
        return record

    def review(self, ticket_id: str, queue: str):
        if queue not in LABELS:
            raise ValueError("Choose a valid queue.")
        ticket = next((x for x in self.tickets if x["id"] == ticket_id), None)
        if ticket is None:
            raise ValueError("Ticket not found.")
        previous = ticket["human_queue"]
        ticket["human_queue"] = queue
        ticket["status"] = "reviewed"
        self.reviews.append({
            "ticket_id": ticket_id, "queue": queue, "previous_human_queue": previous,
            "model_queue": ticket["predicted_queue"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return ticket


_STOPWORDS = set("i my me we our the a an to of and is are it in on for with since please can cannot not has have this all".split())
_ALIASES = {
    "outlook": "email", "mailbox": "email", "mailboxes": "email", "emails": "email",
    "mail": "email", "messages": "email", "message": "email",
    "sending": "send", "sent": "send", "sends": "send",
    "failing": "failure", "failed": "failure", "fails": "failure",
    "stopped": "failure", "broken": "failure", "down": "failure",
}


def ticket_terms(text: str) -> Counter:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return Counter(_ALIASES.get(w, w) for w in words if w not in _STOPWORDS)


def similarity(a: Counter, b: Counter) -> float:
    denominator = math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
    return sum(v * b.get(k, 0) for k, v in a.items()) / denominator if denominator else 0.0


def suspected_incidents(tickets: list[dict], now=None, window_minutes=10,
                        min_tickets=3, similarity_threshold=0.55) -> list[dict]:
    """Complete-link grouping avoids merging dissimilar endpoints through a bridge.

    Lexical similarity with documented email aliases; not root-cause diagnosis.
    Each distinct ticket counts once. Synthetic reporters do not prove real impact.
    """
    if window_minutes <= 0 or min_tickets < 2 or not 0 < similarity_threshold <= 1:
        raise ValueError("Invalid incident grouping settings.")
    if not tickets:
        return []
    times = [datetime.fromisoformat(x["timestamp"]) for x in tickets]
    if any(x.tzinfo is None for x in times):
        raise ValueError("Incident timestamps need a timezone.")
    now = now or max(times)
    if now.tzinfo is None:
        raise ValueError("Current time needs a timezone.")
    recent = {}
    for ticket, time in zip(tickets, times):
        if now - timedelta(minutes=window_minutes) <= time <= now:
            recent[ticket["id"]] = ticket
    groups = []
    for ticket in sorted(recent.values(), key=lambda x: (x["timestamp"], x["id"])):
        terms = ticket_terms(ticket["text"])
        match = next((g for g in groups if all(
            similarity(terms, ticket_terms(other["text"])) >= similarity_threshold
            for other in g)), None)
        if match is None:
            groups.append([ticket])
        else:
            match.append(ticket)
    return [{
        "title": "Suspected shared incident", "ticket_ids": [x["id"] for x in group],
        "count": len(group), "example": group[0]["text"],
        "method": "Lexical similarity with email aliases; complete-link grouping",
        "confirmed": False,
    } for group in groups if len(group) >= min_tickets]
