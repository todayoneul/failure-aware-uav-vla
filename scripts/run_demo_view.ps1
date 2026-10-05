param([ValidateSet('close','medium','elevated')][string]$View='close')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskOutput=Join-Path $taskRoot 'outputs/demo_views'
$taskExe=Join-Path $taskRoot 'assets/projectairsim-blocks-1.0.1/Blocks/Binaries/Win64/Blocks-Win64-Shipping.exe'
$taskPython=Join-Path $taskRoot 'assets/projectairsim-env/Scripts/python.exe'
New-Item -ItemType Directory -Force $taskOutput | Out-Null
if (-not (Test-Path -LiteralPath $taskExe) -or -not (Test-Path -LiteralPath $taskPython)) { throw 'Prepared Blocks and Windows client required; see docs/setup.md' }
if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in 8989,8990) { throw 'Stop the existing simulator/client before this separate model-free demo' }
$taskSim=Start-Process -FilePath $taskExe -WorkingDirectory (Split-Path $taskExe) -ArgumentList @('-windowed','-ResX=1280','-ResY=720','-WinX=0','-WinY=0') -WindowStyle Normal -PassThru
try {
    $taskDeadline=(Get-Date).AddSeconds(35)
    do {
        Start-Sleep -Milliseconds 500
        $taskPorts=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $taskSim.Id -and $_.LocalPort -in 8989,8990 })
    } while ($taskPorts.Count -lt 2 -and (Get-Date) -lt $taskDeadline -and -not $taskSim.HasExited)
    if ($taskPorts.Count -lt 2) { throw 'Simulator ports did not become ready' }
    $taskSim.Id | Set-Content -LiteralPath (Join-Path $taskOutput 'simulator-pid.txt')
    & $taskPython (Join-Path $PSScriptRoot 'capture_demo_views.py') --view $View --window 2>&1 | Tee-Object -FilePath (Join-Path $taskOutput 'capture.log')
    if ($LASTEXITCODE -ne 0) { throw 'Capture failed; inspect outputs/demo_views/capture-result.json' }
} finally {
    $taskRemaining=Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue
    if ($taskRemaining -and $taskRemaining.Path -eq $taskExe) {
        $null=$taskRemaining.CloseMainWindow()
        Start-Sleep -Seconds 2
        if (Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue) { Stop-Process -Id $taskSim.Id }
    }
}
