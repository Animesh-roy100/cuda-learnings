# Download the TinyLlama-1.1B-Chat Q4_0 GGUF that 19-llm-engine runs.
#
#   .\scripts\fetch_model.ps1                      # to $env:USERPROFILE\models
#   .\scripts\fetch_model.ps1 -Dir D:\models
#
# 638 MB, Apache-2.0. Deliberately outside the repository -- and outside
# OneDrive: this repo lives under a OneDrive-synced Desktop, and a model file
# placed there would be uploaded. The engine looks in ~\models by default, or
# wherever $env:CUDA_PORTFOLIO_MODEL points.

param([string]$Dir = (Join-Path $env:USERPROFILE "models"))

$ErrorActionPreference = "Stop"
$name = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
$url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/$name"
$sha256 = "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556"

if ($Dir -match "OneDrive|Dropbox|Google Drive") {
    Write-Warning "$Dir looks like a synced folder; the 638 MB model would be uploaded."
}
New-Item -ItemType Directory -Force $Dir | Out-Null
$dest = Join-Path $Dir $name

if ((Test-Path $dest) -and ((Get-FileHash $dest -Algorithm SHA256).Hash -eq $sha256.ToUpper())) {
    Write-Host "already present and verified: $dest"
    exit 0
}

Write-Host "downloading $name (638 MB) to $Dir"
& curl.exe -fL --retry 3 -o "$dest.part" $url
if ($LASTEXITCODE -ne 0) { throw "download failed" }
$hash = (Get-FileHash "$dest.part" -Algorithm SHA256).Hash
if ($hash -ne $sha256.ToUpper()) {
    Remove-Item "$dest.part"
    throw "SHA-256 mismatch: got $hash"
}
Move-Item "$dest.part" $dest -Force
Write-Host "ok: $dest"
