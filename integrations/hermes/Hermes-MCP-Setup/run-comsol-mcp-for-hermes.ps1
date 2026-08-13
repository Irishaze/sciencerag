$env:JAVA_HOME = "D:\comsol\COMSOL63\Multiphysics\java\win64\jre"
$env:COMSOL_ROOT = "D:\comsol\COMSOL63\Multiphysics"
$env:PIP_CACHE_DIR = "D:\comsol\pip-cache"
$env:TEMP = "D:\comsol\temp"
$env:TMP = "D:\comsol\temp"
$env:HF_HOME = "D:\comsol\hf-cache"
$env:HF_ENDPOINT = "https://hf-mirror.com"

Set-Location "D:\comsol\COMSOL_Multiphysics_MCP"
& "D:\comsol\COMSOL_Multiphysics_MCP\.venv\Scripts\python.exe" -m src.server
