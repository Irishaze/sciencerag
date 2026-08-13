$env:TEC_APP_EXE = "D:\comsol\TEC-Multiphysics-Studio\TEC Multiphysics Studio.exe"
$env:TEC_API_BASE_URL = "auto"
$env:TEC_API_TIMEOUT_SECONDS = "300"
$env:TEC_REPORT_OUTPUT_DIR = "D:\comsol\TEC-Multiphysics-Studio\resources\artifacts\reports"
$env:TEC_AUTO_LAUNCH = "1"

Set-Location "D:\comsol\TEC_Multiphysics_MCP"
& "D:\comsol\COMSOL_Multiphysics_MCP\.venv\Scripts\python.exe" -m src.server
