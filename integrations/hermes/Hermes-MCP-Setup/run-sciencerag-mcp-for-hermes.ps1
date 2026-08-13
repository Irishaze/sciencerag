$env:SCIENCERAG_HOME = "D:\comsol\ScienceRAG"
$env:SCIENCERAG_API_BASE_URL = "auto"
$env:SCIENCERAG_API_TIMEOUT_SECONDS = "180"
$env:SCIENCERAG_AUTO_LAUNCH = "1"

Set-Location "D:\comsol\ScienceRAG_MCP"
& "D:\comsol\COMSOL_Multiphysics_MCP\.venv\Scripts\python.exe" -m src.server
