# Local planner review queue

`generate_planner_review_queue.py` creates paraphrase candidates locally from
already-reviewed seed cases. Each candidate is explicitly marked
`synthetic-unreviewed`; it cannot be used in runtime pattern memory or SFT.

Generate a small batch first:

```powershell
.venv\Scripts\python.exe training\generate_planner_review_queue.py --limit 2 --variants-per-seed 3 --confirm-local-generation
```

For every candidate, a reviewer must verify that its query has exactly the same
intent as `source_case_id`, then change only the following fields:

```json
"review_status": "human-reviewed",
"reviewer": "reviewer-id-or-role",
"review_note": "why the inherited ToolSpec label remains correct"
```

Reject bad candidates by using `rejected`; do not delete audit history. Keep
customer data out of this folder: queries must be anonymized and no source CSV
rows, identifiers, values, credentials, or file paths may be stored here.
