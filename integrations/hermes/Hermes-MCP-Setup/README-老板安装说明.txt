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
1. 把 sciencerag 仓库放到 D:\comsol\ScienceRAG（含 corpus\papers\ 语料，约 735MB，不在 git 里，需单独拷贝；理想情况下把开发机上已经建好的 .pqa_index\ 检索缓存也一起拷过来，否则第一次调用会现场对全部语料重新做 embedding，很慢且消耗 API 额度）。
2. 装好 Python 3.12 + uv。
3. 在 D:\comsol\ScienceRAG 下建 .env（可从 .env.example 复制），填两个 key：
   - DEEPSEEK_API_KEY （LLM，用于文献抽取/回答生成）
   - OPENAI_API_KEY   （embedding，用于检索索引）
4. 右键 PowerShell 运行 D:\comsol\ScienceRAG\integrations\hermes\deploy-sciencerag-windows.ps1
   （sciencerag 仓库整体复制过去后，这个脚本就跟着在里面；也可以单独把它拷到别处跑，
   用 -ScienceragHome 指定 D:\comsol\ScienceRAG）
   —— 会自动检查 uv/.env/语料/索引缓存，跑 `uv sync`，并起一次服务做探活测试。
5. 之后 sciencerag MCP 工具第一次被调用时会自动执行
   `uv run uvicorn sciencerag.app:app --port 8000`（见 run-sciencerag-mcp-for-hermes.ps1），无需手动启动。

注意
- 建议保持 D:\comsol 路径不变，因为 MCP 启动脚本使用该路径。
- 若改路径，需要同步修改 Hermes-MCP-Setup 里的 run-*.ps1。
