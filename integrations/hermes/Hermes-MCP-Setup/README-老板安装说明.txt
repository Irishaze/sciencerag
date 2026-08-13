Hermes + TEC Multiphysics Studio + MCP 交付说明

本包不包含 Hermes 主程序。请老板先在本机安装 Hermes，并至少启动一次生成 config.yaml。

推荐解压位置
1. 将压缩包解压到 D:\，解压后应有：
   D:\comsol\TEC-Multiphysics-Studio
   D:\comsol\TEC_Multiphysics_MCP
   D:\comsol\COMSOL_Multiphysics_MCP
   D:\comsol\ScienceRAG_MCP
   D:\comsol\ScienceRAG        （代码 + 语料 + 检索缓存已包含，仍需老板自己填 .env，见下方"ScienceRAG 部署"）
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
- ScienceRAG：项目代码 + corpus\papers 语料（171 篇 PDF）+ .pqa_index 检索缓存（已建好，首次调用不用现场重新 embedding）+ 已构建好的前端（frontend\dist）。
- Hermes-MCP-Setup：配置脚本、mcp_servers 配置片段、启动脚本（含 run-sciencerag-mcp-for-hermes.ps1、deploy-sciencerag-windows.ps1）。

未包含内容
- Hermes 主程序：老板本机安装。
- 发送方个人账号、API key、auth.json、state.db、sessions、memories。
- TEC 历史大算例缓存 resources\artifacts\mph_cases。
- ScienceRAG 的 .env（DEEPSEEK_API_KEY / OPENAI_API_KEY）：跟其他 API key 一样，出于同样的理由不随包发送，需要老板自己申请、自己填。

ScienceRAG 部署（首次接入需要，之后 Hermes 会自动拉起；代码/语料/索引已经在包里，只差下面这两步）
1. 装好 Python 3.12 + uv。
2. 在 D:\comsol\ScienceRAG 下用 .env.example 建一个 .env，填两个 key：
   - DEEPSEEK_API_KEY （LLM，用于文献抽取/回答生成）
   - OPENAI_API_KEY   （embedding，用于检索索引）
   然后右键 PowerShell 运行 D:\comsol\ScienceRAG\integrations\hermes\deploy-sciencerag-windows.ps1
   —— 会自动检查 uv/.env/语料/索引缓存，跑一次 `uv sync`，并起一次服务做探活测试。
3. 之后 sciencerag MCP 工具第一次被调用时会自动执行
   `uv run uvicorn sciencerag.app:app --port 8000`（见 run-sciencerag-mcp-for-hermes.ps1），无需手动启动。

注意
- 建议保持 D:\comsol 路径不变，因为 MCP 启动脚本使用该路径。
- 若改路径，需要同步修改 Hermes-MCP-Setup 里的 run-*.ps1。
