$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskPython=Join-Path $taskRoot 'assets/projectairsim-env/Scripts/python.exe'
$taskTemp=Join-Path ([IO.Path]::GetTempPath()) ('uav mission '+[guid]::NewGuid().ToString('N'))
try {
    & $taskPython (Join-Path $taskRoot 'scripts/blur_demo_control.py') --action init --mode mission --output $taskTemp --steps 4
    if ($LASTEXITCODE -ne 0) { throw 'Mission initialization failed' }
    $taskControl=Get-Content -LiteralPath (Join-Path $taskTemp 'control.json') -Raw | ConvertFrom-Json
    $taskStatus=Get-Content -LiteralPath (Join-Path $taskTemp 'telemetry.json') -Raw | ConvertFrom-Json
    if ($taskControl.enabled -ne $false -or $taskControl.mission_requests.Count -ne 0 -or $taskControl.overview_view -ne 'top' -or $taskControl.overview_zoom -ne 1) { throw 'Wrong mission control defaults' }
    if ($taskStatus.mission.state -ne 'IDLE' -or $taskStatus.max_steps -ne 4) { throw 'Wrong mission telemetry defaults' }
    foreach ($taskName in @('run_mission_demo.ps1','run_blur_demo.ps1')) {
        $taskTokens=$taskErrors=$null
        $null=[Management.Automation.Language.Parser]::ParseFile((Join-Path $taskRoot "scripts/$taskName"),[ref]$taskTokens,[ref]$taskErrors)
        if ($taskErrors.Count) { throw "Launcher parse failed: $taskName" }
    }
    Write-Output "Mission control / launcher parse PASS on PowerShell $($PSVersionTable.PSVersion)"
} finally {
    foreach ($taskName in @('control.json','telemetry.json','run-info.json')) {
        $taskFile=Join-Path $taskTemp $taskName
        if (Test-Path -LiteralPath $taskFile) { Remove-Item -LiteralPath $taskFile }
    }
    if (Test-Path -LiteralPath $taskTemp) { Remove-Item -LiteralPath $taskTemp }
}
