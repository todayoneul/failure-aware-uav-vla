param([switch]$AutoFailures)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskExe = Join-Path $taskRoot 'assets\projectairsim-blocks-1.0.1\Blocks\Binaries\Win64\Blocks-Win64-Shipping.exe'
$taskPython = Join-Path $taskRoot 'assets\projectairsim-env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskExe) -or -not (Test-Path -LiteralPath $taskPython)) { throw 'Prepared environment required: see docs/setup.md' }
if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in 8989,8990 }) { throw 'Ports 8989/8990 are in use' }
$taskSim = Start-Process -FilePath $taskExe -WorkingDirectory (Split-Path -Parent $taskExe) -ArgumentList @('-windowed','-ResX=960','-ResY=540','-WinX=0','-WinY=40') -WindowStyle Normal -PassThru
try {
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        $listening = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $taskSim.Id -and $_.LocalPort -in 8989,8990 })
    } while ($listening.Count -lt 2 -and (Get-Date) -lt $deadline -and -not $taskSim.HasExited)
    if ($listening.Count -lt 2) { throw 'Simulator did not expose both ports' }
    $taskArgs = @((Join-Path $PSScriptRoot 'projectairsim_probe.py'),'--frames','100','--flight','--sim-pid',"$($taskSim.Id)",'--output','manual-demo-result.json')
    if ($AutoFailures) { $taskArgs += '--auto-failures' }
    & $taskPython @taskArgs
    if ($LASTEXITCODE -ne 0) { throw 'Demo failed: inspect outputs/platform_final/manual-demo-result.json' }
} finally {
    $taskRemaining = Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue
    if ($taskRemaining -and $taskRemaining.Path -eq $taskExe) {
        $null = $taskRemaining.CloseMainWindow()
        Start-Sleep -Seconds 2
        if (Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue) { Stop-Process -Id $taskSim.Id }
    }
}
