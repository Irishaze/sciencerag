param(
  [string]$HermesHome = "D:\Hermes",
  [string]$ComsolRoot = "D:\comsol"
)

$ErrorActionPreference = "Stop"
$configPath = Join-Path $HermesHome "config.yaml"
if (!(Test-Path $configPath)) {
  throw "找不到 Hermes config.yaml：$configPath。请先安装并启动一次 Hermes，或用 -HermesHome 指定安装目录。"
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot "run-tec-mcp-for-hermes.ps1") -Destination (Join-Path $HermesHome "run-tec-mcp-for-hermes.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "run-comsol-mcp-for-hermes.ps1") -Destination (Join-Path $HermesHome "run-comsol-mcp-for-hermes.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "run-sciencerag-mcp-for-hermes.ps1") -Destination (Join-Path $HermesHome "run-sciencerag-mcp-for-hermes.ps1") -Force

$scienceragHome = Join-Path $ComsolRoot "ScienceRAG"
if (!(Test-Path $scienceragHome)) {
  Write-Warning "未找到 $scienceragHome —— ScienceRAG 本体需要单独部署（含 uv sync 依赖、corpus\papers、LLM API key 的 .env）后，sciencerag MCP 工具才能实际调用。这里先只写入 MCP 配置。"
}

$config = Get-Content -LiteralPath $configPath -Raw
if ($config -notmatch "(?m)^mcp_servers:") {
  $config = $config.TrimEnd() + "`r`n" + "mcp_servers:`r`n"
}
if ($config -notmatch "(?m)^  comsol:") {
  $config = $config -replace "(?m)^mcp_servers:\s*$", @"
mcp_servers:
  comsol:
    command: powershell
    args:
      - -NoProfile
      - -ExecutionPolicy
      - Bypass
      - -File
      - D:\Hermes\run-comsol-mcp-for-hermes.ps1
"@
}
if ($config -notmatch "(?m)^  tec:") {
  $config = $config -replace "(?m)^mcp_servers:\s*$", @"
mcp_servers:
  tec:
    command: powershell
    args:
      - -NoProfile
      - -ExecutionPolicy
      - Bypass
      - -File
      - D:\Hermes\run-tec-mcp-for-hermes.ps1
"@
}
if ($config -notmatch "(?m)^  sciencerag:") {
  $config = $config -replace "(?m)^mcp_servers:\s*$", @"
mcp_servers:
  sciencerag:
    command: powershell
    args:
      - -NoProfile
      - -ExecutionPolicy
      - Bypass
      - -File
      - D:\Hermes\run-sciencerag-mcp-for-hermes.ps1
"@
}
Set-Content -LiteralPath $configPath -Encoding UTF8 -Value $config
Write-Host "已写入 Hermes MCP 配置。请重启 Hermes。"
Write-Host "测试口令：调用 tec_control_app，instruction 填：点让 TEC app 切到物理与求解页面，选择研究3，并开始运行求解。"
Write-Host "测试口令（ScienceRAG）：调用 sciencerag_ask，question 填一个 TEC 相关的问题。"
