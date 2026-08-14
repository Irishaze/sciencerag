"""sciencerag.kg_approval — web-panel equivalent of scripts/
approve_kg_candidates.py (spec §7's originally CLI-only "候选 -> 审批 -> 入库"
gate). Same underlying logic either way (sciencerag/validate/kg_approval.py);
this package only adds an HTTP surface over it."""
