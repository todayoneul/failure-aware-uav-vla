$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskPython=Join-Path $taskRoot 'assets/projectairsim-env/Scripts/python.exe'
$taskControl=Join-Path $taskRoot 'scripts/blur_demo_control.py'
$taskTemp=Join-Path ([IO.Path]::GetTempPath()) ('uav blur control '+[guid]::NewGuid().ToString('N'))
try {
    & $taskPython $taskControl --action init --output $taskTemp --steps 6
    if ($LASTEXITCODE -ne 0) { throw 'File-based control initialization failed' }
    $taskState=Get-Content -LiteralPath (Join-Path $taskTemp 'control.json') -Raw | ConvertFrom-Json
    $taskTelemetry=Get-Content -LiteralPath (Join-Path $taskTemp 'telemetry.json') -Raw | ConvertFrom-Json
    if ($taskState.enabled -ne $false -or $taskState.severity -ne 'medium' -or $taskState.quit -ne $false) { throw 'Wrong initial control state' }
    if ($taskTelemetry.max_steps -ne 6) { throw 'Requested step count was lost' }
    & $taskPython $taskControl --action quit --output $taskTemp
    if ($LASTEXITCODE -ne 0) { throw 'File-based quit failed' }
    $taskState=Get-Content -LiteralPath (Join-Path $taskTemp 'control.json') -Raw | ConvertFrom-Json
    if ($taskState.quit -ne $true) { throw 'Quit request not recorded' }
    & $taskPython $taskControl --action quit --output $taskTemp
    if ($LASTEXITCODE -ne 0) { throw 'Repeated quit request failed' }
    Write-Output "Control init/quit PASS on PowerShell $($PSVersionTable.PSVersion)"
} finally {
    # Only files created by this probe; never touch actual demo results.
    foreach ($taskName in @('control.json','control-events.jsonl','telemetry.json','run-info.json')) {
        $taskFile=Join-Path $taskTemp $taskName
        if (Test-Path -LiteralPath $taskFile) { Remove-Item -LiteralPath $taskFile }
    }
    if (Test-Path -LiteralPath $taskTemp) { Remove-Item -LiteralPath $taskTemp }
}
