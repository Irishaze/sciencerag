"""sciencerag.report (spec §5) — audited, citation-backed run reports.

Called after sciencerag.validate. Every quantitative claim in the rendered
report carries an inline citation to a run ID or literature source (spec
§2 "凡论断必有来源"), and each generated report is stored under
data/reports/ keyed by run_id + generated_at so it stays retrievable by run
lineage (spec §5 "报告随运行血缘版本化存储").
"""
