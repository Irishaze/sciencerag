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

- `allow_external`:**M1 阶段是显式 no-op,M6 起是真实功能**——传 `true` 且内部检索覆盖不足时,会真的查 Semantic Scholar(不含 arXiv)。细节见下文「M6」一节。

### 响应

每条 `prior` 包含:

| 字段 | 说明 |
|---|---|
| `kind` | 五选一:`parameter_range`(数值范围)、`material_property`(材料属性)、`scaling_relationship`(标度关系,不一定带数字)、`candidate_config`(候选设计/配置)、`caution`(限制或警告) |
| `value` | 每个 `kind` 有专属固定 schema,不是自由 dict——没有 `summary` 兜底键,抽不出结构化内容就不产出。比如 `parameter_range` 要求 `field_name`(须等于外层 `field`)+ `unit` + 至少一个 `min`/`max`/`typical`;`scaling_relationship` 要求 `x`/`y`(须恰好等于 `related_fields` 的两个参数名)+ `direction`。完整的五套子 schema 见 [docs/spec/sciencerag_spec_zh.md](docs/spec/sciencerag_spec_zh.md) §3.6。`value` 里(以及 `notes` 里)出现的每一个数字都会做**数字溯源校验**:必须能在这条 prior 引用的证据原文里找到(精确匹配或 ≤2% 相对误差,不做单位换算)——找不到就整条拒收重试,详见 §3.8。这是一道确定性硬校验,跟 `confidence` 那个软排序分数是两回事 |
| `confidence` | **一个启发式分数,不是校准过的概率**——公式是 `(来源数量相关的基数) * (证据平均相关性)`,不能解读成"这个结论有 X% 概率正确"。**2026-08-05 起,`confidence` 不再决定一条 prior 进不进最终结果**——三轮独立验证(含一次正式的 4 候选公式对比,noisy-OR/一致性加权/现有公式/LLM 直接打分,全部没测出比现有公式更好的区分度)一致显示它的输入(来源数、相关度)跟先验实际是否可信没有单调关系,所以干脆不再让它把关,只用来对已经通过质量检查的 prior 排序、以及 `max_priors` 超限时决定截断谁。真正把关"这条 prior 对不对"的,是 `value` 那行说的数字溯源校验(确定性)+ 一道独立 LLM 语义判断(`gpt-5.6-luna`,判 KEEP/REVIEW/DROP),详见 [docs/spec/sciencerag_spec_zh.md](docs/spec/sciencerag_spec_zh.md) §3.7/§3.9 |
| `sources` | 每条 prior 至少一个来源(schema 强制 `min_length=1`,不允许无引用的结论)——要么是 `{"type": "paper", "doi": ..., "span": ...}`,要么是 `{"type": "kg_triple", "triple_id": ...}` |

`coverage.gaps` 是人类可读的字符串列表,如实反映这次响应的局限性——比如"证据检索到了但相关性不够,没能抽取"、"抽取出的某些 prior 语义支持性存疑(REVIEW)被排除了,来源 DOI 和具体原因是 XXX"、"某些 prior 通过了所有质量检查但被 max_priors 截断"、"请求了外部检索但 M1 没实现"。**`gaps` 不是错误,是透明度**——即使 `gaps` 非空,响应本身仍然是 `status: "ok"` 的合法结果。真正的失败(检索/LLM 调用异常)走 `status: "error"` 的 `ErrorResponse` 信封,HTTP 502。

### 已知的范围边界(M1 阶段性设计,不是 bug)

- **知识图谱优先分支在 M1 阶段永远为空**(`sciencerag/priors/kg.py`):spec 里 KG 查询优先级最高,但图谱要靠仿真运行积累。M1 阶段图谱确实是空的,自动落到文献检索,是预期行为;M5 起图谱有了真实存储,见下文「M5」一节——只是从零开始积累,冷启动阶段依然经常是空的。
- **延迟是软性护栏,不是硬限制**:目标 30 秒(spec §9),超时只记警告日志 + 审计日志里的 `elapsed_s`,不会让请求失败。

### 测试

```bash
make test-m1-fast   # 免费:单测 + schema 校验 + 路由冒烟测试(mock 掉真实检索)
make test-m1        # 完整:上面的 + 真实调用跑 tests/fixtures/priors_regression.json(有费用,几分钟到十几分钟)
```

回归 fixture 用属性断言(至少几条 prior、必须出现某些 kind、必须有真实 DOI 引用),不是精确文本匹配——LLM 抽取管线的措辞本来就不是逐字稳定的。改了提示词/检索参数/阈值/语料库之后必须重新跑一遍 `make test-m1`(spec §8)。少数几个语料库覆盖薄弱的 fixture 标了 `known_flaky`(根因是 PaperQA2 agent_llm 的检索路径运行间不稳定,试过加 `seed` 没能解决,详见 spec §8),这几条会自动跑 3 次取多数结果,不代表流水线本身不稳。

## `sciencerag.validate` 契约(M2 + 基本版 M3)

`POST /sciencerag/validate` —— 一次仿真跑完之后调用,做两件事:4.1 异常检查(结果在物理上可不可信)、4.2 结果评估(跟已知案例/文献先验对不对得上)。通过检查的结果才会往下走到 M3(4.3 微调建议、4.4 知识候选)。schema 见 [sciencerag/schemas/validate.schema.json](sciencerag/schemas/validate.schema.json)。

### 4.1 异常检查(`sciencerag/validate/checks.py`)

三项检查一起跑:

| 检查项 | 测什么 | 覆盖范围 |
|---|---|---|
| `energy_balance` | 材料交界面(陶瓷-导体-热电臂)两侧的温度/电势/热流/电流是否守恒 | 仅 `n_pairs=1`——多对是拼接出来的虚拟结构,没有真实交界面数据可核对 |
| `pde_residual` | 把预测场代回热电耦合偏微分方程,看残差多大 | `n_pairs` 1–20 都算,但 >1 会标注 `composed_topology_pending_multipair_comsol`(组合预测,未经验证) |
| `ood` | 这次设计的 5 维潜空间坐标 z,离训练时见过的 31 个真实设计有多远(留一法马氏距离) | 需要请求里给 `latent_state`,不给就跳过(记 `info`,不算异常) |

**severity 怎么判**:这两类检查用的是完全不同的判定方式,而且都不是绝对物理阈值——`energy_balance`/`pde_residual` 用相对量:这次残差除以 11 个已知真实解算案例里的最大值,>2 倍记 `warning`,>5 倍记 `blocking`;`ood` 用相对训练集自身分布的百分位:≥90% 记 `warning`,≥99% 记 `blocking`。**这些具体数字(2/5 倍、90/99 百分位)是工程占位符,没有真实数据校准过**,跟 M1 `MIN_EVIDENCE_RELEVANCE` 那种走过完整裁判校准流程的阈值不是一回事。

任意一项 `blocking` → `update_package.blocked=true`,4.2/4.3/4.4 全部短路,不产出评估结论,也不产出微调建议或知识候选。

### 4.2 结果评估(`sciencerag/validate/evaluation.py`)

两条对照线合并成一个 `verdict`:

- `benchmark_comparison`:跟 31 条已知 COMSOL 报告样本比——只有 `design_parameters` 跟某条样本 1% 容差内匹配,才直接比性能数字;对不上就诚实判 `insufficient_benchmark`,不瞎凑。样本库小,大多数新设计目前都会落到这一档。
- `prior_comparison`:跟 M1 检索到的文献先验(`parameter_range` 类型)比,v1 只处理这一种 `kind`,单位不一致直接跳过(不猜换算)。

`verdict` ∈ `consistent` / `deviation_found` / `insufficient_benchmark`。`deviation_found` 不代表谁错了——可能是文献范围本身有偏差、这次设计本来就要突破常规、或者仿真链路有问题,系统只负责把差异摆出来,交给人判断。

### 4.3 微调建议 + 4.4 知识候选(`sciencerag/validate/finetune.py` / `kg_candidates.py`)

前提:`update_package.blocked=false`。

- **微调建议**:两类信号驱动——误差驱动(评估里被判 `deviation` 的条目)、不确定性驱动(warning 级异常)。两个信号都没有就返回 `None`,不硬凑建议。输出是给人审阅的建议(推荐训练样本、损失重加权方向、超参数调整方向),不会自动触发训练。
- **知识候选**:只有 `verdict` 是 `consistent` 或 `insufficient_benchmark` 才抽取(`deviation_found` 的结果"交给人裁断",不当已确认知识)。每条候选的 `confidence` 按 `verdict` 给 0.7/0.4——**同样是启发式,未经校准**。`dedup_status` 靠查询知识图谱判断新/重复/冲突,图谱冷启动阶段基本恒为 `new`。

### 测试

```bash
uv run pytest tests/test_validate_route.py tests/test_validate_schema.py tests/test_validate_regression.py tests/test_validate_m3.py -q
```

`tests/fixtures/validate_regression.json` 是按 spec §8 要求维护的回归集,覆盖"应该 blocking"和"不应误伤"两类案例(不需要真实 API 调用,跑得快)。

## `sciencerag.report` 契约(M4)

`POST /sciencerag/report` —— 验证完成后调用,把这次运行的设计参数、结果、M2 的完整输出(异常、评估、更新提案)、用到的先验,组装成一份带引用的报告。输出 JSON + 渲染好的 Markdown(没有 PDF 渲染器)。报告正文每一条定量论断都带行内引用(运行 ID 或文献 DOI)。

报告按 `run_id + 生成时间` 存到 `data/reports/`(不进 git),供后续按运行血缘浏览:

```
GET /sciencerag/reports            # 列表
GET /sciencerag/reports/{stem}     # 取一份
```

`key_results` 的 `confidence_label` 是定性的(`high` / `check_flagged` / `no_anomaly_data`),不是数值置信区间——tec_surrogate 没有校准过的误差模型可以画出真正的不确定度带,伪造一个数字比不给更误导人。

## `sciencerag.ask` 契约(M5)

`POST /sciencerag/ask` —— 类 MiroFish 模式:先查知识图谱有没有匹配的三元组,有就把子图和问题一起交给 LLM 做有据可依的答案合成;图谱没有匹配,就诚实回退到 M1 的文献检索(`fallback_used`/`coverage_note` 在响应里明确标注,不会不声不响地换一条路)。

知识图谱(`sciencerag/priors/kg.py`)从 M1 阶段的空实现,升级成了真实的 JSON 文件存储(`data/kg/graph.json`):同一 subject/relation/条件的新数据,数值一致就合并来源,数值冲突就并列保留、标记冲突,绝不静默覆盖。**唯一的写入路径**是 `scripts/approve_kg_candidates.py`——读取 M2/M3 产出的知识候选,人工批准后才真正入库(spec §6.3:写入图谱只能走候选→审批这一条路)。

Web 前端有两个:`sciencerag/static/workbench.html`(访问 `/workbench`)是最初的单页静态版本(纯 HTML + vanilla JS,不需要构建),`frontend/` 是后来按 spec §7 补的真正 Vite/React 工程(访问 `/app`,构建产物由后端同进程托管,见下文「前端开发」一节)。两者功能等价——问答面板(含真正的力导向图谱可视化)、报告浏览面板;文献/知识候选审批面板按 spec §7 的明文许可,两个前端都不做,v1 用命令行脚本代替(`scripts/approve_kg_candidates.py`、`scripts/approve_external_papers.py`)。`workbench.html` 保留作为不需要 Node 环境时的轻量参考实现。

## 前端开发(`frontend/`)

spec §7 描述的完整 Vite/React 前端,构建后由 FastAPI 同进程托管(不是独立的 Node 服务;`sciencerag/app.py` 在 `frontend/dist/` 存在时挂载到 `/app`,访问根路径 `/` 会重定向过去)。

```bash
cd frontend
npm install
npm run dev     # 开发模式:http://localhost:5173,API 请求通过 Vite 的 proxy 转给 http://127.0.0.1:8000(需要后端另开一个终端跑 uv run uvicorn sciencerag.app:app --port 8000)
npm run build   # 生产构建 → frontend/dist/,之后后端自己起来就能在 /app 访问,不需要额外的 Node 进程
```

`vite.config.ts` 里 `base: '/app/'` 是关键配置——因为这个前端不是挂在域名根路径,构建产物里的资源引用必须带上 `/app/` 前缀,不然浏览器会去请求不存在的根路径资源(这个坑真的踩过,修的时候还专门加了 `tests/test_static_pages.py` 里两个断言资源路径的测试防止再犯)。

图谱可视化用的是 `react-force-graph-2d`(真正的力导向布局,不是之前 workbench.html 里手搓的圆形排列),报告 Markdown 渲染用 `react-markdown`。

## 外部检索与批量证据(M6)

`allow_external=true` 且内部检索覆盖不足时,`sciencerag/priors/external_retrieval.py` 会去查 Semantic Scholar 的免费搜索 API。范围比 spec §3.5 描述的完整入库流程窄:

- 只取论文摘要当证据文本,不下载解析 PDF 全文。
- 命中的论文标 `provenance: "external_unverified"`,置信度打折扣(0.7 倍,同样是占位值),进入待审队列(`data/external_papers/`),按重复命中次数走 spec §3.5 的"自动转正"规则;期刊白名单规则未实现。
- Semantic Scholar 公开 API 有限流(实测遇到过 429),失败时优雅降级——跳过外部增强、正常返回内部结果,不会让整个 `priors` 请求失败。

批量证据模式(spec §3.4):

```
POST /sciencerag/priors/batch_evidence
```

给定 N 个候选设计,逐一返回支持/反驳/中立的证据分类,不做排序打分——排序机制(如果要的话)留给 Hermes 那边的竞赛/排序逻辑。

## Docker 部署

```bash
docker compose up --build
```

`Dockerfile` 是两阶段构建:第一阶段 `node:22-slim` 编译 `frontend/`(`npm install && npm run build`),第二阶段 `python:3.12-slim` 跑 FastAPI 服务,只把第一阶段的构建产物(`frontend/dist/`)拷进来,Node 本身不进最终镜像。真实 API key 放进 `.env`(参考 `.env.example`),`data/`、`logs/`、PaperQA2 索引通过 volume 持久化,重启容器不丢。torch 锁定 CPU-only 版本(`pyproject.toml` 的 `[tool.uv.sources]`)——这套系统的 torch 只做小图上的推理前向传播,默认版本会额外拉几个 GB 的 NVIDIA CUDA 运行库,完全用不上。

**部署缺口已解决**:`tec_surrogate/` 现在完整提交在本仓库里(排除了两个 120MB 的 COMSOL `.mph` 源文件,超过 GitHub 单文件 100MB 限制,而且部署不需要它们——见 `.gitignore` 里的说明)。已经用真正的全新 `git clone` + `docker build` 验证过端到端能跑通,不是只在本地已填好数据的目录里测过。踩过一个坑记录一下:根目录 `.gitignore` 一开始有条不带前导 `/` 的 `data/` 规则,git 的通配符语义下这条规则会匹配任意深度的 `data/` 目录,结果第一次提交 `tec_surrogate/` 时把 `tec_surrogate/data/`(含运行时要加载的模型文件)整个静默排除了——只有从全新 clone 构建才会暴露这个问题,本地一直"能跑"是因为本地目录本来就有这些文件,不代表 git 里真的有。现在已锚定成 `/data/`,只匹配仓库根目录。

`corpus/papers/` 目前仍只有 6 篇种子论文进了 git,其余靠本地 `.gitignore` 规则管理,不在这次修复范围内(不影响服务能否启动,只影响 `priors` 检索能看到多少论文)。

## 状态

M1–M6(spec §10)均已实现并有真实数据跑通(见各节)。已知限制统一记录在对应小节和代码注释里,不是隐藏的坑;`docs/spec/sciencerag_spec_zh.md` §9 记录了哪些开放问题现在有具体实现可以复核、哪些仍然开放。
