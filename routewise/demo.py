"""A notebook-native desk. No public server or external ticket integrations."""
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
import json
import ipywidgets as w
from IPython.display import display, HTML, clear_output
from .core import Desk, Policy, LABELS, suspected_incidents

SCENARIOS = [
    ("Please install the approved Adobe Acrobat application on my laptop.", "Software"),
    ("I cannot access the shared folder through Teams. Please investigate.", None),
    ("Outlook email sending failed for our team this morning.", "O365"),
    ("Our team email sending failed in Outlook this morning.", "O365"),
    ("Outlook email sending has failed again for our team this morning.", "O365"),
    ("Please decommission the retired test server and remove it from monitoring.", "EOL"),
    ("Please create an Active Directory account for the new employee.", "Active Directory"),
    ("The office printer has a persistent paper jam and needs repair.", "Computer-Services"),
]


def launch_desk(router):
    desk = Desk()
    recorded_mode = bool(getattr(router, 'recorded', False))
    output = w.Output()
    text = w.Textarea(placeholder="Describe the support request…", layout=w.Layout(width="100%", height="90px"))
    threshold = w.FloatSlider(value=.80, min=.15, max=.99, step=.01,
                              description="Score cutoff", style={"description_width": "initial"})
    margin = w.FloatSlider(value=.20, min=0, max=.8, step=.01,
                           description="Top-two gap", style={"description_width": "initial"})
    automatic = w.Checkbox(value=False, description="Enable experimental automatic routing")
    route = w.Button(description="Route ticket", button_style="primary")
    replay = w.Button(description="Replay 8 recorded tickets" if recorded_mode else "Replay 8 synthetic tickets")
    refresh = w.Button(description="Refresh desk")
    export = w.Button(description="Download desk records")
    ticket_select = w.Dropdown(options=[], description="Ticket")
    queue_select = w.Dropdown(options=LABELS, description="Human queue")
    review = w.Button(description="Save human decision")
    notice = w.HTML()

    def policy():
        return Policy(threshold.value, margin.value, automatic.value)

    def table(headers, rows):
        return "<table style='width:100%;border-collapse:collapse'>" + "<tr>" + "".join(
            f"<th style='text-align:left;padding:8px;border-bottom:1px solid #aaa'>{escape(str(h))}</th>"
            for h in headers) + "</tr>" + "".join("<tr>" + "".join(
                f"<td style='padding:8px;border-bottom:1px solid #ddd'>{escape(str(v))}</td>"
                for v in row) + "</tr>" for row in rows) + "</table>"

    def render():
        ticket_select.options = [(f"{r['id']} · {r['status']}", r['id']) for r in desk.tickets]
        with output:
            clear_output(wait=True)
            auto = sum(r["status"] == "automatic" for r in desk.tickets)
            pending = sum(r["status"] == "needs_review" for r in desk.tickets)
            display(HTML(f"<h3>Ticket desk</h3><p>{len(desk.tickets)} tickets · {auto} automatic · "
                         f"{pending} awaiting review · {len(desk.reviews)} human decisions</p>"))
            display(HTML("<p>Scores are relative preferences across seven labels, not validated probabilities "
                         "of correctness. Threshold changes affect new decisions only.</p>"))
            if recorded_mode:
                display(HTML("<p><b>Recorded replay:</b> these are saved scores from the completed GPU run. "
                             "No model inference is executed in this notebook. Only the eight recorded texts are available.</p>"))
            if desk.tickets:
                display(HTML(table(["ID", "Ticket", "Suggested queue", "Status", "Human queue", "Source"], [
                    [r['id'], r['text'], r['predicted_queue'] or 'Inference failed', r['status'],
                     r['human_queue'] or '—', r['source']] for r in desk.tickets])))
                latest = desk.tickets[-1]
                if latest['scores']:
                    display(HTML("<h3>Latest ticket candidate scores</h3>" + table(
                        ["Queue", "Relative score"],
                        [(k, f"{v:.3f}") for k, v in sorted(latest['scores'].items(), key=lambda x:x[1], reverse=True)])))
                display(HTML(f"<p>{escape(latest['reason'])}</p>"))
                incidents = suspected_incidents(desk.tickets)
                display(HTML("<h3>Incident Radar</h3><p>Lexical grouping over a 10-minute window; "
                             "three reports required. Groups are suspected, not confirmed outages.</p>"))
                for group in incidents:
                    display(HTML(f"<p><b>{escape(group['title'])}</b>: {group['count']} reports · "
                                 f"{escape(', '.join(group['ticket_ids']))}<br>{escape(group['example'])}</p>"))
                if not incidents:
                    display(HTML("<p>No group currently meets the incident settings.</p>"))

    def add(ticket, timestamp=None, source="manual"):
        try:
            result = router.score(ticket)
            record = desk.add(ticket, result['scores'], policy(), timestamp, source)
            record.update({k: result[k] for k in ['log_scores', 'latency_ms', 'score_method', 'calibrated']})
            record['inference_mode'] = 'recorded_scores' if recorded_mode else 'live_model'
            record['new_inference_executed'] = not recorded_mode
        except Exception as error:
            # Preserve inference failures as review items, with no invented label or score.
            record = {
                'id': f'T{len(desk.tickets)+1:04d}', 'text': ticket,
                'timestamp': (timestamp or datetime.now(timezone.utc)).isoformat(),
                'source': source, 'scores': {}, 'predicted_queue': None,
                'human_queue': None, 'status': 'needs_review',
                'reason': f'Inference failed: {type(error).__name__}: {error}',
            }
            desk.tickets.append(record)
        return record

    def on_route(_):
        if not text.value.strip():
            notice.value = "Enter a ticket first."
            return
        route.disabled = replay.disabled = True
        notice.value = "Reading saved GPU scores…" if recorded_mode else "Scoring all seven complete labels…"
        try:
            add(text.value)
            render()
            notice.value = "Ticket processed."
        finally:
            route.disabled = replay.disabled = False

    def on_replay(_):
        route.disabled = replay.disabled = True
        start = datetime.now(timezone.utc) - timedelta(minutes=4)
        try:
            for i, (ticket, expected) in enumerate(SCENARIOS):
                notice.value = f"Replaying saved ticket {i+1}/{len(SCENARIOS)}…" if recorded_mode else f"Scoring synthetic ticket {i+1}/{len(SCENARIOS)}…"
                record = add(ticket, start + timedelta(seconds=30*i),
                             'recorded_synthetic_replay' if recorded_mode else 'synthetic_replay')
                record['illustrative_expected_queue'] = expected
                render()
            notice.value = "Synthetic replay complete. These examples are not a benchmark."
        finally:
            route.disabled = replay.disabled = False

    def on_review(_):
        if ticket_select.value:
            desk.review(ticket_select.value, queue_select.value)
            render()
            notice.value = "Human decision saved. Model weights have not changed."

    def on_export(_):
        path = Path('/content/routewise_desk_records.json')
        path.write_text(json.dumps({
            'tickets': desk.tickets, 'reviews': desk.reviews,
            'notice': 'Experimental policy; synthetic replay is not benchmark evidence.',
        }, indent=2), encoding='utf-8')
        from google.colab import files
        files.download(str(path))

    route.on_click(on_route)
    replay.on_click(on_replay)
    refresh.on_click(lambda _: render())
    review.on_click(on_review)
    export.on_click(on_export)
    display(w.VBox([
        w.HTML("<h2>RouteWise Incident Desk</h2><p>" +
               ("Recorded GPU scores · no live model inference" if recorded_mode else "Your restored Qwen LoRA model") +
               " · experimental review policy · notebook session storage</p>"),
        text, w.HBox([threshold, margin]), automatic,
        w.HBox([route, replay, refresh, export]), notice,
        w.HBox([ticket_select, queue_select, review]), output,
    ]))
    render()
    return desk
