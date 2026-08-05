# TEC 潜空间代理模型项目文件说明

## 1. 项目用途

本目录包含 TEC（热电制冷器）代理模型从 COMSOL 数据准备、模型训练、潜空间预测到网页展示的完整工程。

当前主要网页提供：

- 10 个 TEC 设计参数输入；
- 6 个性能指标预测；
- COP 曲线与 COP 热图；
- 5 维潜变量 `z` 的二维投影视图；
- 输入超出训练范围时的警告；
- PN 对数整数校验；
- 1 至 20 对 PN 的温度场、电势场预测接口。

## 2. 当前运行链路

当前版前端和后端共用一个 Python 服务，不需要分别启动两个进程：

```text
web/index.html + web/app.js + web/styles.css
                     |
                     v
prediction_server.py（HTTP 服务和预测 API）
          |                              |
          v                              v
comsol_latent_surrogate.joblib     component_graph_seponet_20pairs.pt
          |                              |
          v                              v
6 项指标、COP、5 维 z               温度场、电势场
```

启动命令（在本目录执行）：

```powershell
..\.venv-deepxde\Scripts\python.exe prediction_server.py --host 127.0.0.1 --port 8765
```

访问地址：`http://127.0.0.1:8765/`

健康检查：`http://127.0.0.1:8765/health`

若该 Python 环境不存在，可使用已安装依赖的 Python：

```powershell
python -m pip install -r requirements.txt
python prediction_server.py --host 127.0.0.1 --port 8765
```

场预测还需要 PyTorch，相关环境和训练命令见 `docs/PHYSICS_FOUNDATION.md`。

## 3. API 和数据格式

### 潜空间/性能预测

- `GET /api/meta`：输入范围、输出名称、COP 网格坐标、训练样本潜空间位置等元数据。
- `POST /api/predict`：接收 10 个设计参数并返回预测结果。

返回值中的 `latent` 就是最终 5 维潜在状态 `z`。它是 JSON 数值数组，反序列化后为 JavaScript `number[]` 或 Python `list[float]`，不是文件路径、内存地址或引用位置：

```json
{
  "latent": [z1, z2, z3, z4, z5],
  "latent_2d": [z1, z2]
}
```

`latent_2d` 只用于网页散点图显示；完整可复用状态应使用 `latent`。后端对 `z` 依次执行 PCA 逆变换和输出缩放逆变换，解码结果为：

```json
{
  "scalars": {
    "delta_T_max_K": 0.0,
    "optimal_current_A": 0.0,
    "optimal_voltage_V": 0.0,
    "total_resistance_ohm": 0.0,
    "max_heat_dissipation_W": 0.0,
    "figure_of_merit_1_per_K": 0.0
  },
  "cop_surface": [[0.0]],
  "outside_training_range": []
}
```

当前 `cop_surface` 的形状为 `3 x 8`，行对应 3 个温差，列对应 8 个工作电流。具体坐标由 `/api/meta` 的 `delta_t_values` 和 `currents` 给出。

### 场预测

- `GET /api/field-meta`：场模型元数据。
- `POST /api/field-predict`：根据 `n_pairs`、`current_A`、`hot_temperature_K` 返回温度场和电势场。

## 4. 主要文件

### 当前网页和服务

| 路径 | 用途 |
| --- | --- |
| `web/index.html` | 当前版潜空间网页结构 |
| `web/app.js` | 输入校验、API 请求、指标渲染、COP 曲线/热图和潜空间绘图 |
| `web/styles.css` | 当前版网页样式和响应式布局 |
| `prediction_server.py` | 当前 HTTP 服务、静态文件服务、性能预测和场预测 API |
| `tests/test_prediction_server.py` | 当前预测服务测试，覆盖 10 输入、6 输出、5 维 z、COP 形状、范围警告和 PN 整数校验 |

### 当前模型和数据

| 路径 | 用途 |
| --- | --- |
| `data/models/comsol_latent_surrogate.joblib` | 当前 COMSOL 潜空间代理模型 |
| `data/processed/comsol_report_dataset.npz` | 当前训练数值数据集 |
| `data/processed/comsol_report_dataset.json` | 当前数据集元信息 |
| `outputs/comsol_latent_training.json` | 潜空间模型训练摘要和验证指标 |
| `outputs/figures/comsol_latent_training.png` | 潜空间模型训练结果图 |
| `outputs/component_graph_seponet_20pairs.pt` | 当前 1 至 20 PN 对场模型权重 |
| `outputs/component_graph_seponet_20pairs.json` | 当前场模型训练摘要 |
| `data/component_cases/tec_1pair_dset3.npz` | 单 PN 对 COMSOL 场参考数据 |

### 场模型代码

| 路径 | 用途 |
| --- | --- |
| `physics_foundation/__init__.py` | 场模型包导出入口 |
| `physics_foundation/graph.py` | 变尺寸组件图和批处理数据结构 |
| `physics_foundation/model.py` | 组件图网络和局部 SepONet 解码器 |
| `physics_foundation/losses.py` | 监督、界面守恒和物理残差损失 |
| `physics_foundation/comsol_export.py` | COMSOL 组件和场数据导出 |
| `physics_foundation/real_data.py` | 真实 COMSOL 数据读取与转换 |
| `physics_foundation/synthetic.py` | 合成图数据和架构验证数据 |
| `physics_foundation/multi_pair.py` | 多 PN 对分层图构造和组合预测 |
| `physics_foundation/field_service.py` | 网页使用的温度场/电势场推理服务 |
| `docs/PHYSICS_FOUNDATION.md` | 场模型原理、现状、训练命令和限制说明 |

### 数据处理和训练脚本

| 路径 | 用途 |
| --- | --- |
| `scripts/01_simplify_model.py` | 简化 COMSOL 模型 |
| `scripts/02_probe_expressions.py` | 探测可用 COMSOL 表达式 |
| `scripts/03_define_probes.py` | 定义输出探针 |
| `scripts/04_sobol_design.py` | 生成 Sobol 参数设计和数据划分 |
| `scripts/05_run_batch.py` | 批量运行 COMSOL 仿真，可选导出场数据 |
| `scripts/06_verify_pipeline.py` | 验证数据生成流水线 |
| `scripts/07_pca_analysis.py` | 对输出进行 PCA 分析 |
| `scripts/08_toy_surrogate.py` | 训练和演示早期玩具代理模型 |
| `scripts/09_train_from_reports.py` | 从 COMSOL 报告训练当前潜空间代理模型 |
| `scripts/10_generate_comsol_training_batch.py` | 生成 COMSOL 训练批次 |
| `scripts/11_train_component_graph_demo.py` | 训练合成组件图/SepONet 演示模型 |
| `scripts/12_export_comsol_component_case.py` | 导出已求解的单 PN 对 COMSOL 组件场数据 |
| `scripts/13_train_real_component_graph.py` | 用真实单 PN 对场数据训练组件图模型 |
| `scripts/14_train_multipair_component_graph.py` | 训练 1 至 20 PN 对组合场模型 |
| `legacy/build_standalone_predictor.py` | 构建旧版单文件预测网页 |
| `config.py` | COMSOL、路径和参数配置 |

### 测试文件

| 路径 | 用途 |
| --- | --- |
| `tests/test_prediction_server.py` | 当前潜空间预测服务测试 |
| `tests/test_train_from_reports.py` | COMSOL 报告解析和训练测试 |
| `tests/test_generate_comsol_training_batch.py` | COMSOL 训练批次生成测试 |
| `tests/test_comsol_component_export.py` | 组件场导出测试 |
| `tests/test_physics_foundation.py` | 场模型基础结构和损失测试 |
| `tests/test_multipair_foundation.py` | 多 PN 对模型和推理测试 |

### 依赖、配置和 COMSOL 模型

| 路径 | 用途 |
| --- | --- |
| `requirements.txt` | 潜空间训练和网页服务基础依赖 |
| `requirements-physics-foundation.txt` | 场模型/PyTorch 相关依赖 |
| `models/study_info.json` | COMSOL study 信息 |
| `models/tec_1pair.mph` | 单 PN 对 COMSOL 原始模型 |
| `models/tec_1pair_working.mph` | 单 PN 对 COMSOL 工作模型 |

### 历史版本

以下文件不是当前 `prediction_server.py + web/` 运行链路的首选入口，保留作回溯或离线演示：

| 路径 | 用途 |
| --- | --- |
| `legacy/server.py` | 旧版服务入口 |
| `legacy/web/predict_ui.html` | 旧版预测网页 |
| `legacy/web/tec_latent_predictor_standalone.html` | 旧版独立单文件预测器 |
| `legacy/models/surrogate_v1.joblib` | 旧版代理模型 |
| `legacy/models/toy_surrogate.joblib` | 玩具代理模型 |

## 5. 完整目录清单

下面列出项目有效文件；`__pycache__/` 和 `*.pyc` 等运行缓存未列入。

```text
tec_surrogate/
|-- config.py
|-- prediction_server.py
|-- README.md
|-- requirements.txt
|-- requirements-physics-foundation.txt
|-- web/
|   |-- index.html
|   |-- app.js
|   `-- styles.css
|-- scripts/
|   |-- 01_simplify_model.py
|   |-- 02_probe_expressions.py
|   |-- 03_define_probes.py
|   |-- 04_sobol_design.py
|   |-- 05_run_batch.py
|   |-- 06_verify_pipeline.py
|   |-- 07_pca_analysis.py
|   |-- 08_toy_surrogate.py
|   |-- 09_train_from_reports.py
|   |-- 10_generate_comsol_training_batch.py
|   |-- 11_train_component_graph_demo.py
|   |-- 12_export_comsol_component_case.py
|   |-- 13_train_real_component_graph.py
|   `-- 14_train_multipair_component_graph.py
|-- tests/
|   |-- __init__.py
|   |-- test_comsol_component_export.py
|   |-- test_generate_comsol_training_batch.py
|   |-- test_multipair_foundation.py
|   |-- test_physics_foundation.py
|   |-- test_prediction_server.py
|   `-- test_train_from_reports.py
|-- docs/
|   `-- PHYSICS_FOUNDATION.md
|-- legacy/
|   |-- README.md
|   |-- server.py
|   |-- build_standalone_predictor.py
|   |-- models/
|   |   |-- surrogate_v1.joblib
|   |   `-- toy_surrogate.joblib
|   `-- web/
|       |-- predict_ui.html
|       `-- tec_latent_predictor_standalone.html
|-- physics_foundation/
|   |-- __init__.py
|   |-- comsol_export.py
|   |-- field_service.py
|   |-- graph.py
|   |-- losses.py
|   |-- model.py
|   |-- multi_pair.py
|   |-- real_data.py
|   `-- synthetic.py
|-- models/
|   |-- study_info.json
|   |-- tec_1pair.mph
|   `-- tec_1pair_working.mph
|-- data/
|   |-- component_cases/tec_1pair_dset3.npz
|   |-- models/comsol_latent_surrogate.joblib
|   |-- processed/comsol_report_dataset.json
|   |-- processed/comsol_report_dataset.npz
|   |-- processed/dataset.json
|   |-- processed/dataset_v2.json
|   |-- raw/sample_0000.npz ... sample_0006.npz
|   |-- raw/doc/sample_02.docx ... sample_34.docx
|   |-- raw/doc/thermoelectric_cooler.docx
|   |-- raw/doc/Thermoelectric_Cooler_COMSOL_Design_1.docx
|   |-- probe_definitions.npz
|   |-- region_bboxes_nominal.npz
|   |-- simulation_status.db
|   |-- sobol_256_norm.npy
|   |-- sobol_256_phys.npy
|   |-- sobol_metadata.json
|   |-- split_calibration_norm.npy
|   |-- split_calibration_phys.npy
|   |-- split_test_norm.npy
|   |-- split_test_phys.npy
|   |-- split_train_norm.npy
|   |-- split_train_phys.npy
|   |-- split_val_norm.npy
|   `-- split_val_phys.npy
`-- outputs/
    |-- batch_reports/batch_status.csv
    |-- figures/*.png
    |-- logs/*.log
    |-- component_graph_seponet_20pairs.json
    |-- component_graph_seponet_20pairs.pt
    |-- component_graph_seponet_real.json
    |-- component_graph_seponet_real.pt
    |-- comsol_latent_training.json
    |-- comsol_popup_test.csv
    |-- comsol_training_batch_50.csv
    |-- COMSOL_TRAINING_BATCH_50.md
    |-- comsol_training_batch_50_summary.json
    |-- expression_report.json
    |-- model_params.json
    |-- pca_results.json
    |-- sobol_50_samples.csv
    `-- tec_1pair_component_export.json
```

## 6. 数据和模型限制

- 性能代理模型依据现有 COMSOL 报告样本训练，超出训练输入范围的结果属于外推，网页会给出警告。
- 场模型的单 PN 对结果直接以 COMSOL 数据为锚点；2 至 20 PN 对目前是组合预测，不能等同于多 PN 对 COMSOL 精度验证。
- 5 维 `z` 是经过训练得到的压缩表示，各维没有预先指定的单一物理含义；应通过解码后的指标、COP 面或敏感性分析解释，不能把 `z1` 至 `z5` 直接当作温度、电流等物理参数。
- 模型文件、训练时的 scaler 和 PCA 必须配套使用，不能只拿 `z` 脱离当前解码器直接还原物理量。
