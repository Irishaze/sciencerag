# ScienceRAG

ScienceRAG 是 TEC(热电制冷)仿真闭环架构中的科学 RAG 服务,为 Hermes 智能体提供四个端点:

- `sciencerag.priors` — 实验前的先验检索
- `sciencerag.validate` — 仿真运行后的验证与学习
- `sciencerag.report` — 报告生成
- `sciencerag.ask` — 知识图谱问答

完整设计规格见 [docs/spec/sciencerag_spec_zh.md](docs/spec/sciencerag_spec_zh.md)。

## 开发环境

依赖 Python 3.12,用 [uv](https://docs.astral.sh/uv/) 管理:

```bash
uv sync
```

## 状态

当前处于 M1(`sciencerag.priors`,仅内部文献)开发阶段。
