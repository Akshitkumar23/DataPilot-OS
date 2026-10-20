# DataPilot research candidate human review

This workspace contains 630 AI-assisted candidates. It does not contain human approvals.

Complete every row in `variant_review_template.csv`. Each review dimension must be `pass`, `fail`, or `not_applicable`. Use `approved` only when no dimension fails; otherwise use `rejected` or `needs_changes`.

A reviewer must independently check language naturalness, semantic intent, method, artifacts, expected clarification/unsupported behavior, dependency edges and safety. Fill reviewer, reviewer_role, rationale, independent_expected_result and a timezone-aware reviewed_at_utc.

The manifest binds each displayed record to the frozen corpus. Do not edit query or contract columns. Run `training/import_human_reviews.py` only after completing every row. Human approval still does not make abstract placeholder candidates executable SFT records; approved archetypes must next be instantiated against real dataset schemas and verified by execution oracles.
