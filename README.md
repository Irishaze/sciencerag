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

## `sciencerag.priors` 契约(M1)

`POST /sciencerag/priors` —— 给定一个研究问题,从内部文献库(`corpus/papers/`)检索证据,用 LLM 抽取成结构化的"先验"(priors)。请求/响应的完整 schema 冻结在 [sciencerag/schemas/priors.schema.json](sciencerag/schemas/priors.schema.json)(每次改 `sciencerag/priors/models.py` 后跑 `uv run python scripts/export_schemas.py` 重新导出)。

### 请求

```json
{
  "query": "What is the optimal driving voltage for a thermoelectric cooler to maximize COP?",
  "task_context": {"objective": "...", "constraints": {"heat_load_w": 5.0}},
  "max_priors": 5,
  "allow_external": false
}
```

- `allow_external`:**M1 内是显式 no-op**(spec §9 OQ#1)。传 `true` 不会真的查外部文献(Semantic Scholar/arXiv),只会在响应的 `coverage.gaps` 里加一条提示,说明外部检索要到 M6 才实现。这是刻意的范围控制,不是遗漏——原因和排期见 [docs/spec/sciencerag_spec_zh.md](docs/spec/sciencerag_spec_zh.md) §9、§10。

### 响应

每条 `prior` 包含:

| 字段 | 说明 |
|---|---|
| `kind` | 五选一:`parameter_range`(数值范围)、`material_property`(材料属性)、`scaling_relationship`(标度关系,不一定带数字)、`candidate_config`(候选设计/配置)、`caution`(限制或警告) |
| `value` | 每个 `kind` 有专属固定 schema,不是自由 dict——没有 `summary` 兜底键,抽不出结构化内容就不产出。比如 `parameter_range` 要求 `field_name`(须等于外层 `field`)+ `unit` + 至少一个 `min`/`max`/`typical`;`scaling_relationship` 要求 `x`/`y`(须恰好等于 `related_fields` 的两个参数名)+ `direction`。完整的五套子 schema 见 [docs/spec/sciencerag_spec_zh.md](docs/spec/sciencerag_spec_zh.md) §3.6 |
| `confidence` | **一个启发式分数,不是校准过的概率**——公式是 `(来源数量相关的基数) * (证据平均相关性)`,用来做"值不值得展示"的排序和过滤,不能解读成"这个结论有 X% 概率正确"。下游消费时应该按相对高低排序使用,不要做校准意义上的数值解读。 |
| `sources` | 每条 prior 至少一个来源(schema 强制 `min_length=1`,不允许无引用的结论)——要么是 `{"type": "paper", "doi": ..., "span": ...}`,要么是 `{"type": "kg_triple", "triple_id": ...}` |

`coverage.gaps` 是人类可读的字符串列表,如实反映这次响应的局限性——比如"证据检索到了但相关性不够,没能抽取"、"抽取出的某些 prior 置信度太低被过滤掉了,来源 DOI 是 XXX"、"请求了外部检索但 M1 没实现"。**`gaps` 不是错误,是透明度**——即使 `gaps` 非空,响应本身仍然是 `status: "ok"` 的合法结果。真正的失败(检索/LLM 调用异常)走 `status: "error"` 的 `ErrorResponse` 信封,HTTP 502。

### 已知的范围边界(M1 阶段性设计,不是 bug)

- **知识图谱优先分支是空实现**(`sciencerag/priors/kg.py`):spec 里 KG 查询优先级最高,但图谱要靠仿真运行积累,M1 阶段永远查询为空,自动落到文献检索,这是预期行为。
- **`allow_external` 是 no-op**,见上文。
- **延迟是软性护栏,不是硬限制**:目标 30 秒(spec §9),超时只记警告日志 + 审计日志里的 `elapsed_s`,不会让请求失败。

### 测试

```bash
make test-m1-fast   # 免费:单测 + schema 校验 + 路由冒烟测试(mock 掉真实检索)
make test-m1        # 完整:上面的 + 真实调用跑 tests/fixtures/priors_regression.json(有费用,几分钟到十几分钟)
```

回归 fixture 用属性断言(至少几条 prior、必须出现某些 kind、必须有真实 DOI 引用),不是精确文本匹配——LLM 抽取管线的措辞本来就不是逐字稳定的。改了提示词/检索参数/阈值/语料库之后必须重新跑一遍 `make test-m1`(spec §8)。

## 状态

当前处于 M1(`sciencerag.priors`,仅内部文献)开发阶段。
