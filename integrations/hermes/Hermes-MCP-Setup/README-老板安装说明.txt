Hermes + TEC Multiphysics Studio + MCP 交付说明

本包不包含 Hermes 主程序。请老板先在本机安装 Hermes，并至少启动一次生成 config.yaml。

推荐解压位置
1. 将压缩包解压到 D:\，解压后应有：
   D:\comsol\TEC-Multiphysics-Studio
   D:\comsol\TEC_Multiphysics_MCP
   D:\comsol\COMSOL_Multiphysics_MCP
   D:\comsol\ScienceRAG_MCP
   D:\comsol\ScienceRAG        （单独部署，见下方"ScienceRAG 部署"）
   D:\Hermes-MCP-Setup

2. 安装/连接 MCP：
   右键 PowerShell 运行：
   D:\Hermes-MCP-Setup\Install-MCP-For-Hermes.ps1

   如果 Hermes 不在 D:\Hermes，则运行：
   powershell -ExecutionPolicy Bypass -File D:\Hermes-MCP-Setup\Install-MCP-For-Hermes.ps1 -HermesHome "你的Hermes目录"

3. 重启 Hermes。

4. 测试口令：
   调用 tec_control_app，instruction 填：点让 TEC app 切到物理与求解页面，选择研究3，并开始运行求解。
   调用 sciencerag_ask，question 填一个 TEC 相关的问题（需先完成下方"ScienceRAG 部署"）。

已包含内容
- TEC Multiphysics Studio：已包含当前优化后的 app 与运行时。
- TEC_Multiphysics_MCP：Hermes 到 TEC 可见 app/后端的 MCP 适配器。
- COMSOL_Multiphysics_MCP：COMSOL/多物理接口 MCP 依赖与 Python 环境。
- ScienceRAG_MCP：Hermes 到 ScienceRAG（问答/文献先验/校验/报告/知识图谱审批）后端的 MCP 适配器，与 TEC_Multiphysics_MCP 同一形状，复用 COMSOL_Multiphysics_MCP 的 .venv 运行。
- Hermes-MCP-Setup：配置脚本、mcp_servers 配置片段、启动脚本（含 run-sciencerag-mcp-for-hermes.ps1）。

未包含内容
- Hermes 主程序：老板本机安装。
- 发送方个人账号、API key、auth.json、state.db、sessions、memories。
- TEC 历史大算例缓存 resources\artifacts\mph_cases。
- ScienceRAG 本体（代码仓库、corpus\papers 语料、LLM/embedding API key 的 .env）：与 TEC/COMSOL 不同，ScienceRAG 目前不是离线打包好的应用，需要单独部署到 D:\comsol\ScienceRAG 才能让 sciencerag_* 工具实际可用，否则 Install 脚本仍会写入 MCP 配置，但调用时会报 API 不可达。

ScienceRAG 部署（首次接入需要，之后 Hermes 会自动拉起）
1. 把 sciencerag 仓库放到 D:\comsol\ScienceRAG。
2. 安装 Python 3.12 + uv，在该目录下运行一次 `uv sync`。
3. 把 corpus\papers\ 语料和 .env（LLM/embedding API key）放好。
4. 之后 sciencerag MCP 工具第一次被调用时会自动执行
   `uv run uvicorn sciencerag.app:app --port 8000`（见 run-sciencerag-mcp-for-hermes.ps1），无需手动启动。

注意
- 建议保持 D:\comsol 路径不变，因为 MCP 启动脚本使用该路径。
- 若改路径，需要同步修改 Hermes-MCP-Setup 里的 run-*.ps1。
