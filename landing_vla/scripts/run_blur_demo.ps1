param([ValidateRange(1,400)][int]$MaxSteps=30,[switch]$AutoTest,[string]$Distro='Ubuntu',[ValidateSet('blur','mission')][string]$Mode='blur',[ValidateSet('','step','continuous')][string]$Flight='',[ValidateSet('legacy','grounding-film')][string]$Policy='legacy',[string]$Checkpoint='',[Nullable[double]]$StartX=$null,[Nullable[double]]$StartY=$null,[Nullable[double]]$StartYaw=$null,[Nullable[double]]$StartHeight=$null,[string]$Layout='')
$ErrorActionPreference='Stop'
# -Policy grounding-film: the same Mission Control, flown by an AeroVLA-OFT checkpoint (docs/grounding_film_interactive_demo.md).
$taskGrounding=($Mode -eq 'mission' -and $Policy -eq 'grounding-film')
if ($Policy -ne 'legacy' -and $Mode -ne 'mission') { throw '-Policy grounding-film needs -Mode mission' }
if (-not $taskGrounding -and $MaxSteps -gt 60) { throw 'MaxSteps above 60 is for -Policy grounding-film' }
if ($Checkpoint -and -not $taskGrounding) { throw '-Checkpoint is for -Policy grounding-film' }
# -StartX / -StartY / -StartYaw / -StartHeight: where the first mission starts, instead of the map's first preset launch
# (metres in the map's frame, degrees with 0 facing +x and 90 facing +y, metres above the ground). -Layout: another layout
# of the interactive map (mission_cap puts a landing cap on the orange ball). docs/arbitrary_start_generalized_landing.md.
$taskStartArgs=@()
foreach ($taskPair in @(@('--start-x',$StartX),@('--start-y',$StartY),@('--start-yaw',$StartYaw),@('--start-height',$StartHeight))) {
    if ($null -ne $taskPair[1]) { $taskStartArgs+=$taskPair[0]; $taskStartArgs+=([double]$taskPair[1]).ToString('R',[Globalization.CultureInfo]::InvariantCulture) }
}
if ($Layout) { $taskStartArgs+='--layout'; $taskStartArgs+=$Layout }
if ($taskStartArgs.Count -and -not $taskGrounding) { throw '-StartX, -StartY, -StartYaw, -StartHeight and -Layout are for -Policy grounding-film' }
if ($Mode -eq 'mission' -and $AutoTest) { throw 'AutoTest is blur-only; use scripts/run_model_evaluation.ps1 for mission evaluation' }
# The mission demo flies continuously whichever launcher starts it; -Flight step restores stop-and-go.
if ($Mode -eq 'mission' -and -not $Flight) { $Flight='continuous' }
. (Join-Path $PSScriptRoot 'native_process_args.ps1')
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskOutputRelative=if($taskGrounding){'outputs/mission_demo_grounding'}elseif($Mode -eq 'mission'){'outputs/mission_demo'}else{'outputs/failure_demo'}
$taskOutput=Join-Path $taskRoot $taskOutputRelative
$taskExe=Join-Path $taskRoot 'assets/projectairsim-blocks-1.0.1/Blocks/Binaries/Win64/Blocks-Win64-Shipping.exe'
$taskPython=Join-Path $taskRoot 'assets/projectairsim-env/Scripts/python.exe'
$taskControlScript=Join-Path $PSScriptRoot 'blur_demo_control.py'
if (-not (Test-Path -LiteralPath $taskExe) -or -not (Test-Path -LiteralPath $taskPython)) { throw 'Prepared Blocks and Windows client required; see docs/setup.md' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'outputs/integration/model-downloads.json'))) { throw 'Local model manifest required; see docs/setup.md. This launcher never downloads models.' }
if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in 8989,8990) { throw 'Close the existing simulator/client before starting this demo.' }
$taskWslRoot=& wsl -d $Distro --exec wslpath -u $taskRoot.Replace('\','/')
if ($LASTEXITCODE -ne 0 -or -not $taskWslRoot) { throw 'WSL repository path unavailable' }
$taskWslRoot=$taskWslRoot.Trim()
$taskWslHome=(& wsl -d $Distro --exec printenv HOME).Trim()
$taskWslPython="$taskWslHome/uav-vla-smoke/integration/bin/python"
& wsl -d $Distro --exec test -x $taskWslPython
if ($LASTEXITCODE -ne 0) { throw 'Prepared WSL integration env required; see docs/setup.md' }
$taskHost=((& wsl -d $Distro --exec ip -4 route show default) -split '\s+')[2]
if (-not $taskHost) { throw 'Current Windows NAT host address unavailable' }
New-Item -ItemType Directory -Force $taskOutput | Out-Null
if ($AutoTest) { $MaxSteps=6 }
& $taskPython $taskControlScript --action init --output $taskOutput --steps $MaxSteps --mode $Mode --policy $Policy
if ($LASTEXITCODE -ne 0) { throw 'Demo control initialization failed; previous evidence is preserved in runs/ when archived' }
if ($taskGrounding) {
    # The checkpoint is checked before anything starts; the frozen baseline must still match its recorded fingerprints.
    $taskCheckArgs=@((Join-Path $PSScriptRoot 'mission_checkpoint.py'),'--output',$taskOutput)
    if ($Checkpoint) { $taskCheckArgs+='--checkpoint'; $taskCheckArgs+=$Checkpoint }
    & $taskPython @taskCheckArgs | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint check failed; see the lines above' }
    $taskCheck=Get-Content -LiteralPath (Join-Path $taskOutput 'checkpoint-check.json') -Raw | ConvertFrom-Json
    $taskCheckpoint=$taskCheck.checkpoint
    # A start given here is held to the rule every start has to meet before the simulator is started.
    $taskStartCheck=@((Join-Path $PSScriptRoot 'mission_start.py'),'--output',$taskOutput)+$taskStartArgs
    & $taskPython @taskStartCheck | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw 'INVALID START: a flight may not start there; see the lines above' }
}
$taskRunToken=(Get-Content -LiteralPath (Join-Path $taskOutput 'run-info.json') -Raw | ConvertFrom-Json).id
$taskSim=$taskMonitor=$taskViewer=$taskWorker=$null
$taskCleanupFailure=$null
try {
    if ($taskGrounding) {
        Write-Host 'GROUNDING-FILM MISSION CONTROL'
        Write-Host ''
        Write-Host '1. Click a semantic object or press N.'
        Write-Host '2. Press T to choose LAND / APPROACH.'
        Write-Host '3. Press G to start. Every mission starts from the start pose (L: next preset, S: place one on the map).'
        Write-Host '4. Front/Down show actual model inputs.'
        Write-Host '5. Target XYZ/distance/bearing are evaluator-only.'
        Write-Host '6. R resets to the same start pose, with the target kept.'
        Write-Host '7. Q/Esc safely exits.'
        Write-Host ''
        Write-Host 'Start pose (S):'
        Write-Host 'map click = position | drag = heading | [ ] = height | J K = yaw | S again = done'
        Write-Host ''
        Write-Host 'Landing surfaces:'
        Write-Host 'LAND on a pad, a cube or a cylinder; a sphere or a cone has none (APPROACH only).'
        Write-Host ''
        Write-Host 'Model:'
        Write-Host "$(Split-Path -Leaf $taskCheckpoint) ($($taskCheck.name), grounding $($taskCheck.grounding), frozen baseline $($taskCheck.frozen.state))"
        Write-Host ''
        Write-Host 'Failure injection:'
        Write-Host 'OFF'
        Write-Host ''
    }
    elseif ($Mode -eq 'mission') { Write-Host "MISSION CONTROL | map click or N landmark | G start | M prompt mode | R reset | B blur | Q/Esc abort and land | flight: $Flight" }
    else { Write-Host 'Gaussian Blur Demo | B: toggle | 1/2/3: severity | Q/Esc: exit' }
    Write-Host 'Click the observer window, or use its buttons. Keys apply at the next observation.'
    $taskSim=Start-Process -FilePath $taskExe -WorkingDirectory (Split-Path $taskExe) -ArgumentList @('-windowed','-ResX=1280','-ResY=720','-WinX=0','-WinY=0') -WindowStyle Normal -PassThru
    $taskDeadline=(Get-Date).AddSeconds(35)
    do {
        Start-Sleep -Milliseconds 500
        $taskPorts=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $taskSim.Id -and $_.LocalPort -in 8989,8990 })
    } while ($taskPorts.Count -lt 2 -and (Get-Date) -lt $taskDeadline -and -not $taskSim.HasExited)
    if ($taskPorts.Count -lt 2) { throw 'Simulator ports did not become ready' }
    $taskPS=(Get-Process -Id $PID).Path
    $taskMonitorArguments=Join-NativeArguments @('-NoProfile','-File',(Join-Path $PSScriptRoot 'communication_final_resources.ps1'),'-SimulatorProcessId',"$($taskSim.Id)",'-OutputDirectory',$taskOutputRelative)
    $taskMonitor=Start-Process -FilePath $taskPS -ArgumentList $taskMonitorArguments -WindowStyle Hidden -PassThru
    $taskViewerFile=if($Mode -eq 'mission'){'mission_viewer.py'}else{'blur_demo_viewer.py'}
    $taskViewerArgs=@((Join-Path $PSScriptRoot $taskViewerFile))
    if ($taskGrounding) { $taskViewerArgs+='--output'; $taskViewerArgs+=$taskOutputRelative }
    if ($AutoTest) { $taskViewerArgs+='--auto-close' }
    $taskViewer=Start-Process -FilePath $taskPython -WorkingDirectory $taskRoot -ArgumentList (Join-NativeArguments $taskViewerArgs) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'viewer.log') -RedirectStandardError (Join-Path $taskOutput 'viewer-errors.log')
    $taskRunnerFile=if($taskGrounding){'src/mission/oft_runner.py'}elseif($Mode -eq 'mission'){'src/mission/runner.py'}else{'src/integration/closed_loop_runner.py'}
    $taskRunnerFlag=if($Mode -eq 'mission'){'--mission-demo'}else{'--blur-demo'}
    $taskRunnerArgs=@('-d',$Distro,'--exec',$taskWslPython,"$taskWslRoot/$taskRunnerFile",'--host',$taskHost,$taskRunnerFlag,'--steps',"$MaxSteps",'--run-token',$taskRunToken)
    if ($AutoTest) { $taskRunnerArgs+='--auto-test' }
    if ($taskGrounding) {
        $taskRunnerArgs+='--checkpoint'
        $taskRunnerArgs+=if([IO.Path]::IsPathRooted($taskCheckpoint)){(& wsl -d $Distro --exec wslpath -u $taskCheckpoint.Replace('\','/')).Trim()}else{"$taskWslRoot/$taskCheckpoint"}
        $taskRunnerArgs+=$taskStartArgs
    }
    elseif ($Mode -eq 'mission' -and $Flight) { $taskRunnerArgs+='--flight'; $taskRunnerArgs+=$Flight }
    $taskWorker=Start-Process -FilePath 'wsl.exe' -ArgumentList (Join-NativeArguments $taskRunnerArgs) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'worker.log') -RedirectStandardError (Join-Path $taskOutput 'worker-errors.log')
    Enable-ProcessExitTracking $taskWorker
    if($taskGrounding){Write-Host 'The map loads first and the vehicle takes off to its start pose; the checkpoint loads beside it.'}
    elseif($Mode -eq 'mission'){Write-Host 'Map loads first. Select a surface and press G; the cached model loads on the first mission.'}
    else { Write-Host 'WSL is loading the existing AeroVLA checkpoint. Progress is shown in the observer and worker.log.' }
    $taskPrevious=''
    $taskQuitDeadline=$null
    while (-not $taskWorker.HasExited) {
        if ($taskViewer.HasExited) {
            if (-not $taskQuitDeadline) { $taskQuitDeadline=(Get-Date).AddSeconds(45) }
            & $taskPython $taskControlScript --action quit --output $taskOutput
        }
        try {
            $taskStatus=Get-Content -LiteralPath (Join-Path $taskOutput 'telemetry.json') -Raw | ConvertFrom-Json
            $taskProgress="Step $($taskStatus.step)/$MaxSteps | $($taskStatus.phase)"
            # An AeroVLA-OFT mission decides twice a second: its state is reported, not every decision.
            if ($taskGrounding) { $taskProgress=if($taskStatus.mission.state -eq 'NAVIGATING'){"Mission $($taskStatus.mission.mission_id) flying"}else{"$($taskStatus.mission.state) | $($taskStatus.phase)"} }
            if ($taskProgress -ne $taskPrevious) { Write-Host $taskProgress; $taskPrevious=$taskProgress }
            $taskControl=Get-Content -LiteralPath (Join-Path $taskOutput 'control.json') -Raw | ConvertFrom-Json
            if ($taskControl.quit) { $taskQuitDeadline=if ($taskQuitDeadline) { $taskQuitDeadline } else { (Get-Date).AddSeconds(45) } }
        } catch {}
        if ($taskQuitDeadline -and (Get-Date) -ge $taskQuitDeadline) { break }
        Start-Sleep -Milliseconds 200
        $taskWorker.Refresh(); $taskViewer.Refresh()
    }
    if (-not $taskWorker.HasExited) { throw 'Graceful stop timed out; ownership-aware cleanup will run' }
    if ($null -eq $taskWorker.ExitCode -or $taskWorker.ExitCode -ne 0) { throw "WSL demo failed (exit $($taskWorker.ExitCode)); inspect $taskOutputRelative/worker-errors.log and telemetry.json" }
    if ($AutoTest) { $null=$taskViewer.WaitForExit(10000) }
    else {
        Write-Host 'Run complete. Inspect the last input/action in the observer; press Q/Esc to close.'
        while (-not $taskViewer.HasExited) { Start-Sleep -Milliseconds 200; $taskViewer.Refresh() }
    }
} finally {
    if ($taskWorker -and -not $taskWorker.HasExited) {
        & $taskPython $taskControlScript --action quit --output $taskOutput
        $taskGraceMs=if ($taskQuitDeadline) { [Math]::Max(0,[int](($taskQuitDeadline-(Get-Date)).TotalMilliseconds)) } else { 45000 }
        $null=$taskWorker.WaitForExit($taskGraceMs)
    }
    if ($taskWorker) {
        try {
            $taskCleanup=& wsl -d $Distro --exec $taskWslPython "$taskWslRoot/scripts/stop_blur_worker.py" --pid-file "$taskWslRoot/$taskOutputRelative/worker-pid.txt" --run-token $taskRunToken --mode $(if($taskGrounding){'grounding'}else{$Mode})
            if ($LASTEXITCODE -ne 0) { throw 'Owned Linux worker shutdown could not be verified' }
            $taskCleanup | Set-Content -LiteralPath (Join-Path $taskOutput 'launcher-cleanup.json') -Encoding utf8
            if (-not $taskWorker.WaitForExit(5000)) { Stop-Process -Id $taskWorker.Id }
        } catch { $taskCleanupFailure=$_.Exception.Message }
    }
    if ($taskSim) {
        $taskRemaining=Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue
        if ($taskRemaining -and $taskRemaining.Path -eq $taskExe) {
            $null=$taskRemaining.CloseMainWindow(); Start-Sleep -Seconds 2
            if (Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue) { Stop-Process -Id $taskSim.Id }
        }
    }
    foreach ($taskOwned in @($taskMonitor,$taskViewer)) {
        if ($taskOwned -and -not $taskOwned.HasExited) { Stop-Process -Id $taskOwned.Id }
    }
    if ($taskCleanupFailure) { throw $taskCleanupFailure }
}
