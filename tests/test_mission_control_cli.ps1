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
    foreach ($taskName in @('run_mission_demo.ps1','run_blur_demo.ps1','run_grounding_film_mission_demo.ps1','run_surface_flights.ps1')) {
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
    # A start of the user's own: every launcher takes it, the control file knows the start mode, and a start that may not be
    # flown is refused by the check the launcher runs before it starts the simulator.
    if ($taskControl.start_mode -ne $false) { throw 'The grounding-film control does not begin outside start mode' }
    foreach ($taskName in @('run_mission_demo.ps1','run_blur_demo.ps1','run_grounding_film_mission_demo.ps1')) {
        $taskParameters=(Get-Command (Join-Path $taskRoot "scripts/$taskName")).Parameters
        foreach ($taskParameter in @('StartX','StartY','StartYaw','StartHeight','Layout')) {
            if (-not $taskParameters.ContainsKey($taskParameter)) { throw "$taskName has no -$taskParameter" }
        }
        if ($taskParameters['StartX'].ParameterType -ne [Nullable[double]]) { throw "$taskName does not take -StartX as an optional number" }
    }
    $taskSaved=$ErrorActionPreference; $ErrorActionPreference='Continue'
    $taskLines=& $taskPython (Join-Path $taskRoot 'scripts/mission_start.py') --output $taskTemp --start-x 18 --start-y 72 --start-yaw 30 --start-height 4
    $taskValid=$LASTEXITCODE
    $taskCheck=Get-Content -LiteralPath (Join-Path $taskTemp 'start-check.json') -Raw | ConvertFrom-Json
    $taskLines=& $taskPython (Join-Path $taskRoot 'scripts/mission_start.py') --start-x 10 --start-y 92
    $taskInside=$LASTEXITCODE
    $taskLines=& $taskPython (Join-Path $taskRoot 'scripts/mission_start.py') --start-height 20
    $taskHigh=$LASTEXITCODE
    $taskLines=& $taskPython (Join-Path $taskRoot 'scripts/mission_start.py') --start-x -45.6 --start-y 12 --layout mission_cap
    $taskOutside=$LASTEXITCODE
    $ErrorActionPreference=$taskSaved
    if ($taskValid -ne 0 -or -not $taskCheck.valid -or $taskCheck.start.x -ne 18 -or $taskCheck.start.yaw_deg -ne 30 -or $taskCheck.start.height_m -ne 4) { throw 'A start on open ground was not accepted as given' }
    if ($taskInside -eq 0 -or $taskHigh -eq 0 -or $taskOutside -eq 0) { throw 'A start inside a pad, above the ceiling or outside the map was accepted' }
    Write-Output "Mission control / launcher parse PASS on PowerShell $($PSVersionTable.PSVersion)"
} finally {
    # A second initialization archives the first one's files under runs/; the whole temporary folder is this test's own.
    if (Test-Path -LiteralPath $taskTemp) { Remove-Item -LiteralPath $taskTemp -Recurse }
}
