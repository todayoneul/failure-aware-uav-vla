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
    foreach ($taskName in @('run_mission_demo.ps1','run_blur_demo.ps1','run_grounding_film_mission_demo.ps1')) {
        $taskTokens=$taskErrors=$null
        $null=[Management.Automation.Language.Parser]::ParseFile((Join-Path $taskRoot "scripts/$taskName"),[ref]$taskTokens,[ref]$taskErrors)
        if ($taskErrors.Count) { throw "Launcher parse failed: $taskName" }
    }
    # The AeroVLA-OFT policy: its own defaults, a longer step budget, and the earlier policy left as it was.
    & $taskPython (Join-Path $taskRoot 'scripts/blur_demo_control.py') --action init --mode mission --policy grounding-film --output $taskTemp --steps 320
    if ($LASTEXITCODE -ne 0) { throw 'Grounding-film mission initialization failed' }
    $taskControl=Get-Content -LiteralPath (Join-Path $taskTemp 'control.json') -Raw | ConvertFrom-Json
    $taskStatus=Get-Content -LiteralPath (Join-Path $taskTemp 'telemetry.json') -Raw | ConvertFrom-Json
    if ($taskControl.policy -ne 'grounding-film' -or $taskControl.prompt_mode -ne 'instruction-only' -or $taskControl.direction_hint -ne $false -or $taskControl.enabled -ne $false) { throw 'Wrong grounding-film control defaults' }
    if ($taskStatus.policy.mode -ne 'grounding-film' -or $taskStatus.max_steps -ne 320 -or $taskStatus.mission.state -ne 'IDLE') { throw 'Wrong grounding-film telemetry defaults' }
    $taskSaved=$ErrorActionPreference; $ErrorActionPreference='Continue'
    & $taskPython (Join-Path $taskRoot 'scripts/blur_demo_control.py') --action init --mode mission --output $taskTemp --steps 320 2>$null
    $taskRefused=$LASTEXITCODE; $ErrorActionPreference=$taskSaved
    if ($taskRefused -eq 0) { throw 'The legacy policy accepted more than 60 steps' }
    $taskDefault=(Get-Command (Join-Path $taskRoot 'scripts/run_mission_demo.ps1')).Parameters['Policy'].Attributes | Where-Object { $_ -is [System.Management.Automation.ValidateSetAttribute] }
    if (($taskDefault.ValidValues -join ',') -ne 'legacy,grounding-film') { throw 'run_mission_demo.ps1 lost its -Policy choices' }
    Write-Output "Mission control / launcher parse PASS on PowerShell $($PSVersionTable.PSVersion)"
} finally {
    # A second initialization archives the first one's files under runs/; the whole temporary folder is this test's own.
    if (Test-Path -LiteralPath $taskTemp) { Remove-Item -LiteralPath $taskTemp -Recurse }
}
