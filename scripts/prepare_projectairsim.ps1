$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOutput = Join-Path $taskRoot 'outputs\platform_final'
$archivePath = Join-Path $taskRoot 'assets\archives\ProjectAirSim-Blocks-Windows-1.0.1.zip'
$environmentPath = Join-Path $taskRoot 'assets\projectairsim-blocks-1.0.1'
New-Item -ItemType Directory -Force -Path (Split-Path $archivePath),$taskOutput | Out-Null
$release = Get-Content -LiteralPath (Join-Path $taskRoot 'configs/projectairsim-release.json') -Raw | ConvertFrom-Json
$asset = $release.assets | Where-Object name -eq 'Blocks-Windows-1.0.1.zip'
if ($release.tag_name -ne 'v1.0.1' -or $asset.size -gt 1GB) { throw 'Unexpected release or oversized asset' }
if (-not (Test-Path -LiteralPath $archivePath)) {
    & curl.exe --fail --location --retry 1 --output $archivePath $asset.browser_download_url
    if ($LASTEXITCODE -ne 0) { throw 'Official environment download failed' }
}
$actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ((Get-Item -LiteralPath $archivePath).Length -ne $asset.size -or "sha256:$actualHash" -ne $asset.digest) { throw 'Archive size/SHA256 mismatch' }
if (-not (Test-Path -LiteralPath $environmentPath)) { Expand-Archive -LiteralPath $archivePath -DestinationPath $environmentPath }
[ordered]@{
    url=$asset.browser_download_url; release=$release.tag_name; bytes=$asset.size
    sha256=$actualHash; extraction=$environmentPath
    executables=@(Get-ChildItem -LiteralPath $environmentPath -Recurse -Filter '*.exe' | ForEach-Object FullName)
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskOutput 'download.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $taskOutput 'download.json')
