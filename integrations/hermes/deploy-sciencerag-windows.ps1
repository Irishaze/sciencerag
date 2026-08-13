param(
  [string]$ScienceragHome = "D:\comsol\ScienceRAG",
  [switch]$SkipSync
)

# Provisions ScienceRAG on the target Windows machine so ScienceRAG_MCP's
# auto-launch (`uv run uvicorn sciencerag.app:app --port 8000`) actually has
# something to talk to. Run this AFTER copying/cloning the sciencerag repo to
# $ScienceragHome (not included in this script — the repo itself, corpus
# PDFs, and .pqa_index are large and move separately, e.g. via the same
# Quark drive as everything else in this delivery).

$ErrorActionPreference = "Stop"

if (!(Test-Path $ScienceragHome)) {
  throw "找不到 $ScienceragHome —— 请先把 sciencerag 仓库（含 corpus\papers 和理想情况下 .pqa_index）复制到这里，再跑本脚本。"
}
Set-Location $ScienceragHome

Write-Host "== 1. 检查 uv =="
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
  throw "找不到 uv。先装：powershell -ExecutionPolicy Bypass -Command `"irm https://astral.sh/uv/install.ps1 | iex`"，然后重开终端再跑本脚本。"
}
uv --version

Write-Host "== 2. 检查 .env =="
$envPath = Join-Path $ScienceragHome ".env"
if (!(Test-Path $envPath)) {
  $examplePath = Join-Path $ScienceragHome ".env.example"
  if (Test-Path $examplePath) {
    Copy-Item -LiteralPath $examplePath -Destination $envPath
    Write-Warning "已从 .env.example 生成 .env，但里面的 DEEPSEEK_API_KEY / OPENAI_API_KEY 还是空的 —— 必须手动填好这两个 key，sciencerag.priors/.ask 才能真正调用 LLM 和 embedding。"
  } else {
    throw "既没有 .env 也没有 .env.example，无法继续 —— 需要 DEEPSEEK_API_KEY（LLM）和 OPENAI_API_KEY（embedding）。"
  }
} else {
  $envContent = Get-Content -LiteralPath $envPath -Raw
  if ($envContent -match "DEEPSEEK_API_KEY=\s*(\r?\n|$)" -or $envContent -match "OPENAI_API_KEY=\s*(\r?\n|$)") {
    Write-Warning ".env 里 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 看起来是空的 —— 请确认填好了真实 key。"
  }
}

Write-Host "== 3. 检查语料 corpus\papers =="
$corpusPath = Join-Path $ScienceragHome "corpus\papers"
$pdfCount = 0
if (Test-Path $corpusPath) {
  $pdfCount = (Get-ChildItem -LiteralPath $corpusPath -Filter "*.pdf" -ErrorAction SilentlyContinue).Count
}
if ($pdfCount -eq 0) {
  Write-Warning "corpus\papers 下没有找到 PDF —— sciencerag.priors/.ask 的检索会查不到任何文献。开发机上这个目录大约 735MB，需要单独复制过来（不在 git 里）。"
} else {
  Write-Host "找到 $pdfCount 个 PDF。"
}

Write-Host "== 4. 检查检索索引缓存 .pqa_index =="
$indexPath = Join-Path $ScienceragHome ".pqa_index"
if (!(Test-Path $indexPath)) {
  Write-Warning ".pqa_index 不存在 —— 第一次调用 sciencerag_priors/sciencerag_ask 时会现场对整个语料库重新做 embedding，耗时长且会消耗 OPENAI_API_KEY 额度。如果开发机上已经建好索引，建议直接把 .pqa_index 文件夹复制过来，跳过这一步。"
} else {
  Write-Host ".pqa_index 已存在，检索会直接复用缓存。"
}

if (-not $SkipSync) {
  Write-Host "== 5. uv sync（安装依赖：fastapi/paper-qa/torch/sentence-transformers 等，体积较大，可能需要几分钟） =="
  uv sync
} else {
  Write-Host "== 5. 跳过 uv sync（-SkipSync） =="
}

Write-Host "== 6. 冒烟测试：起服务 + 探活 =="
$proc = Start-Process -FilePath "uv" -ArgumentList "run","uvicorn","sciencerag.app:app","--port","8000" -PassThru -WindowStyle Hidden
try {
  $ready = $false
  for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
      $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/sciencerag/reports" -UseBasicParsing -TimeoutSec 2
      if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
  }
  if ($ready) {
    Write-Host "ScienceRAG API 已就绪：http://127.0.0.1:8000"
  } else {
    Write-Warning "30 秒内没探测到 ScienceRAG API 就绪，检查上面几步是否都通过，或看 uv run uvicorn 的报错。"
  }
} finally {
  Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "部署检查完成。Hermes 侧不需要手动再启动 ScienceRAG —— run-sciencerag-mcp-for-hermes.ps1 会在第一次调用 sciencerag_* 工具时自动执行 'uv run uvicorn sciencerag.app:app --port 8000'。"
