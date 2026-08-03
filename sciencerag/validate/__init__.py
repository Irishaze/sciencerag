"""sciencerag.validate (spec §4) — anomaly checks + result evaluation.

M2 scope only (spec §10): 4.1 anomaly checks + 4.2 result evaluation. 4.3
(fine-tune suggestions) and 4.4 (KG candidates) are M3 — this module always
returns an empty, non-blocking-shaped `update_package` for those two fields.
"""
