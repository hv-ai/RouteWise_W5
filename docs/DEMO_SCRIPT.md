# RouteWise three minute demonstration

## Opening about twenty seconds

"I fine-tuned a small Qwen model to route IT support tickets. I then built RouteWise to
show the operational decisions around that model: when to automate, when to request
human review, and when related reports might point to one shared incident."

Show the original completed assignment and the corrected baseline comparison briefly.

## State which demonstration is running

If using notebook 03 say: "I already ran the model on a Colab T4. This recording replays
those saved model scores because my GPU quota is exhausted. The repository also includes
the live-model notebook. This replay does not run inference on new tickets."

Do not describe the recorded notebook as a currently running model.

## Replay about fifty seconds

Set the score cutoff to 0.90 and gap to 0.20, enable the experimental policy, and replay
the eight recorded tickets once. Show four automatic routes and four tickets awaiting review.

"The shared-folder-through-Teams request is ambiguous. RouteWise exposes the competing
queues and requests review. The score is a relative model preference, not a guarantee."

## Human decision about thirty seconds

Choose T0008, Computer-Services, and Save human decision. Show the original prediction
and the human confirmation together, with one saved decision and three pending requests.

"The model suggested the right queue, but its score was below our experimental cutoff.
A reviewer confirms it. The decision is logged; clicking this button does not retrain the model."

## Incident Radar about thirty seconds

Scroll to the suspected group containing T0003, T0004, and T0005.

"These three email reports form a suspected shared incident, including one ticket still
awaiting review. The grouping uses words and a few email aliases. It does not diagnose a root cause."

## Evidence and limitations about fifty seconds

"On 117 previously inspected development tickets, the tuned model correctly routed 82:
70.1%. With the same full-label scoring method the base model correctly routed 28: 23.9%.
A fixed TF-IDF classifier achieved 69.2%, so I am not claiming a meaningful advantage over
that simpler model."

"At the exploratory 0.90 cutoff, 38 tickets were accepted and 35 were correct: 92.1%
accuracy among automatic routes, at 32.5% coverage. Seventy-nine tickets required review,
and three accepted routes were wrong. The threshold was examined on this same dataset;
this is not independent test performance."

"Free generation also produced unwanted trailing text, which is preserved in the results.
The desk instead scores the seven complete routing labels. My next experiment separates
training, calibration, and testing, but that extra run was blocked by GPU limits."

End by showing the repository's code, executed notebooks, measured results, and tests.
The video should demonstrate what works and state what remains unverified.
