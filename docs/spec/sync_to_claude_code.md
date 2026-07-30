# 同步:先验 output 对齐仿真参数契约(sim_params.json)

## 背景 / 为什么改
经与 mentor 确认了仿真侧(COMSOL 单级 Bi2Te3 TEC)的真实参数后,发现当前
`PriorsResponse` 的 `field` 命名和仿真参数表对不齐(例如我们用 `leg_length_um`,
仿真里叫 `leg_length`,单位 mm)。Hermes 拿先验去填仿真时需要做名称/单位翻译,
易出错。M2 的先验对比(按 field 匹配仿真结果数值)也会因此对不上。

因此引入一份"仿真参数契约" `sim_params.json`,让先验的 field 直接对齐仿真参数名。

## 重要:不改什么(请勿动这些)
- **`kind` 枚举五个值保持不变**:parameter_range / material_property /
  scaling_relationship / candidate_config / caution。**不要增删或改名。**
- `PriorsResponse` / `PriorsRequest` 的整体 schema 结构不变(status/priors/
  coverage/trace_id 等)。
- confidence 公式框架不变。
- PaperQA2 检索层不变。
- 错误信封、trace_id、审计日志基础设施不变。

**这次改的是 `field` 的取值来源 + 抽取逻辑 + 新增一个可选字段 related_fields,
不是 kind,也不是 schema 整体结构。**

## 要改什么

### 1. 新增契约文件 sim_params.json
放入仓库(建议 `sciencerag/priors/sim_params.json` 或 `schemas/` 下)。内容另附
(已定稿, contract_version 1.1.0)。要点:
- `geometry_free`: 12 个几何自由参数(leg_length/leg_width/pitch/d_conductor/
  d_ceramics/length/width/height/sink_base_h/sink_fin_h/sink_fin_w/sink_fin_n),
  每个带 name(仿真真实名) + unit。**这是先验 field 唯一允许的取值来源。**
- `material`: Bi2Te3 固定,含 200-400K 温度表。**prior_target=false,先验不碰材料。**
- `operating_condition` / `derived` / `numerical_setting`: 均 prior_target=false,
  先验不涉及。

### 2. Prior 模型新增可选字段 related_fields(支持关系型先验)
- 背景:scaling_relationship 类先验表达的是"两个/多个参数之间的关系"(如
  leg_length 的最优值依赖 leg_width),用单数 field 无法表达"一对参数"。
- 新增字段:`related_fields: list[str] = []`(可选,默认空列表)。
  - 单参数先验(parameter_range/material_property/candidate_config/caution):
    照旧用 `field`,不填 related_fields。
  - 关系型先验(scaling_relationship):`field` 可留空或标主参数,用
    `related_fields` 列出关系涉及的参数,如 ["leg_length","leg_width"]。
- **这是向后兼容的小扩展**:field 不变,related_fields 是新增可选字段,
  老的单参数先验完全不受影响。
- 校验:related_fields 里每个元素也必须 ∈ 契约 geometry_free 的 name 集合。

### 3. 抽取逻辑(extract.py)改为"目标导向抽取"
- 从"开放抽取(LLM 自由起 field 名)"改为"针对契约 geometry_free 里的 12 个参数抽取"。
- prompt 里把这 12 个参数名+单位作为目标清单给 LLM,要求:
  - 只对这些参数出先验,field/related_fields 必须用契约里的确切参数名(不是自造名)。
  - **不预设参数分组、不假设参数间关系**:参数关系(如耦合)只有当证据里
    实际提到时才作为 scaling_relationship 抽出(填 related_fields);
    文献没提就不造。
  - 忠实报告文献实际说了什么,抽不到的参数不硬凑。

### 4. field / related_fields 校验(硬约束)
- 抽取后校验:先验的 field(若有)和 related_fields 里每个元素,都必须
  ∈ 契约 geometry_free 的 name 集合。
- 不在集合内的一律拒绝(防止 LLM 自造名或误碰材料/派生参数)。
- 对应契约 rules.prior_field_must_be_in_contract。

### 5. gaps 生成:用契约当标尺,并带归因
- 契约里 12 个几何参数,本次抽取覆盖了哪些、没覆盖哪些 → 没覆盖的进 gaps。
- **gaps 要带原因归因**,区分三种情况(不要笼统写"没覆盖"):
  - 未检索到相关证据 → "文献中未检索到 <param> 相关证据"
  - 检索到但相关性不足(<MIN_EVIDENCE_RELEVANCE) → "检索到 <param> 证据但相关性不足"
  - 抽到先验但置信度不足(<CONFIDENCE_THRESHOLD) → "<param> 提取到先验但置信度不足"
  - (第一版可先简化为"未覆盖",归因分类可后续迭代)

### 6. 注意 top-k / 覆盖率(需评估,不一定这次改)
- 检索目标从"回答单 query"变成"覆盖 12 个参数",现有 top-k 可能偏小,
  导致部分参数证据根本没被取进来 → 产生"假 gap"(文献其实有,只是没检索到)。
- 建议用回归 probe 数据评估:不同 k 下 12 个参数的覆盖率,找拐点后再定 k。
- 这一步是评估性的,先记录 TODO,不要盲目调大 k。

## 五种 kind 在新契约体系下的填法(实现时严格照此)

前提:field 与 related_fields 的取值都必须来自契约 geometry_free.name。
kind 枚举本身不变,变的只是 field 怎么填 + value 结构。

### 1) parameter_range —— 单个几何参数的取值范围(最主力)
```json
{
  "kind": "parameter_range",
  "field": "leg_length",
  "value": {"min": 0.02, "max": 0.2, "typical": 0.06, "unit": "mm"},
  "confidence": 0.82,
  "sources": [{"type": "paper", "doi": "10.xxxx/...", "span": "p.4, Fig.3"}]
}
```
- field = 一个契约参数名。value 给 min/max/typical(+单位,与契约一致)。
- 不填 related_fields。

### 2) material_property —— 材料属性
```json
{
  "kind": "material_property",
  "field": null,
  "value": {"note": "材料属性由契约 sim_params.json 固定登记,不由先验提供"},
  "confidence": ...,
  "sources": [...]
}
```
- **注意**:本项目材料固定 Bi2Te3、属性为已知常量(prior_target=false),
  正常流程下**不应产出 material_property 先验**。抽取时不针对材料出先验。
- 保留这个枚举值只为 schema 兼容;若 LLM 抽出材料类先验,应被过滤/不采纳。
- (即:这个 kind 在当前项目实际上是"存在于枚举但不产出"的状态。)

### 3) scaling_relationship —— 参数间关系(用 related_fields)
```json
{
  "kind": "scaling_relationship",
  "field": null,
  "related_fields": ["leg_length", "leg_width"],
  "value": {"form": "coupled", "description": "臂长最优值依赖臂宽,长细比影响电热平衡"},
  "confidence": 0.7,
  "sources": [{"type": "paper", "doi": "10.xxxx/...", "span": "p.5, Eq.12"}],
  "notes": "文献中臂长最优值随臂宽变化,建议联合扫描而非各自独立取最优"
}
```
- **这是 related_fields 的主要使用场景。** 单变量的标度关系(如"COP 随
  leg_length 非单调")related_fields 可只含一个参数 ["leg_length"]。
- 关系涉及谁,由文献决定,不预设。

### 4) candidate_config —— 一组现成的配置(多个参数打包)
```json
{
  "kind": "candidate_config",
  "field": null,
  "related_fields": ["leg_length", "leg_width", "pitch"],
  "value": {"leg_length": 0.07, "leg_width": 0.12, "pitch": 0.05, "reported_cop": 1.2, "unit": "mm"},
  "confidence": 0.6,
  "sources": [{"type": "paper", "doi": "10.xxxx/...", "span": "p.9, Sec.5"}],
  "notes": "文献中一组已报道设计,可作扫描起点"
}
```
- candidate_config 天然涉及多个参数,用 related_fields 列出这组配置涉及哪些参数;
  value 里给每个参数的具体值(key 用契约参数名)。

### 5) caution —— 注意事项/负面知识
```json
{
  "kind": "caution",
  "field": "leg_length",
  "value": {"regime": "below_0.05mm_dominant", "issue": "接触电阻主导"},
  "confidence": 0.85,
  "sources": [{"type": "paper", "doi": "10.xxxx/...", "span": "p.3"}],
  "notes": "臂长低于约 50um 时接触电阻成为主导损耗,仿真忽略将高估 COP"
}
```
- caution 通常挂在某个具体参数上,用 field。若涉及多参数,也可用 related_fields。

## 边界 / 待定(不要自行假设)
- 材料属性数值已从 mentor 处拿到(TEClozicxuse.mph, 200-400K 温度表),已填入契约。
- 材料"适用温度范围外拒绝"的策略暂未加(讨论过但决定先不做),不要自行加。
- sim_params.json 是否放进 schemas/、是否像 priors.schema.json 一样导出/版本化,
  按现有 export_schemas.py 模式处理,保持一致。

## 验收
- 先验 field / related_fields 全部来自契约 geometry_free.name;非法值被拒(加单测)。
- kind 枚举未改动(回归测试应仍全绿)。
- related_fields 为向后兼容可选字段;单参数先验不受影响(加单测)。
- 文献未覆盖的几何参数进 gaps 而非硬凑(加测试:构造只覆盖部分参数的场景)。
- 材料类先验不产出(material 固定,prior_target=false)。
- 现有 M1 回归测试(make test-m1)保持全绿。
