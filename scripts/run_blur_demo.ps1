param([ValidateRange(1,60)][int]$MaxSteps=30,[switch]$AutoTest,[string]$Distro='Ubuntu',[ValidateSet('blur','mission')][string]$Mode='blur')
$ErrorActionPreference='Stop'
if ($Mode -eq 'mission' -and $AutoTest) { throw 'AutoTest is blur-only; use scripts/run_model_evaluation.ps1 for mission evaluation' }
. (Join-Path $PSScriptRoot 'native_process_args.ps1')
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskOutputRelative=if($Mode -eq 'mission'){'outputs/mission_demo'}else{'outputs/failure_demo'}
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
& $taskPython $taskControlScript --action init --output $taskOutput --steps $MaxSteps --mode $Mode
if ($LASTEXITCODE -ne 0) { throw 'Demo control initialization failed; previous evidence is preserved in runs/ when archived' }
$taskRunToken=(Get-Content -LiteralPath (Join-Path $taskOutput 'run-info.json') -Raw | ConvertFrom-Json).id
$taskSim=$taskMonitor=$taskViewer=$taskWorker=$null
$taskCleanupFailure=$null
try {
    if ($Mode -eq 'mission') { Write-Host "MISSION CONTROL | map click or N landmark | G start | M prompt mode | R reset | B blur | Q/Esc abort and land" }
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
    if ($AutoTest) { $taskViewerArgs+='--auto-close' }
    $taskViewer=Start-Process -FilePath $taskPython -WorkingDirectory $taskRoot -ArgumentList (Join-NativeArguments $taskViewerArgs) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'viewer.log') -RedirectStandardError (Join-Path $taskOutput 'viewer-errors.log')
    $taskRunnerFile=if($Mode -eq 'mission'){'src/mission/runner.py'}else{'src/integration/closed_loop_runner.py'}
    $taskRunnerFlag=if($Mode -eq 'mission'){'--mission-demo'}else{'--blur-demo'}
    $taskRunnerArgs=@('-d',$Distro,'--exec',$taskWslPython,"$taskWslRoot/$taskRunnerFile",'--host',$taskHost,$taskRunnerFlag,'--steps',"$MaxSteps",'--run-token',$taskRunToken)
    if ($AutoTest) { $taskRunnerArgs+='--auto-test' }
    $taskWorker=Start-Process -FilePath 'wsl.exe' -ArgumentList (Join-NativeArguments $taskRunnerArgs) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'worker.log') -RedirectStandardError (Join-Path $taskOutput 'worker-errors.log')
    Enable-ProcessExitTracking $taskWorker
    if($Mode -eq 'mission'){Write-Host 'Map loads first. Select a surface and press G; the cached model loads on the first mission.'}
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
            $taskCleanup=& wsl -d $Distro --exec $taskWslPython "$taskWslRoot/scripts/stop_blur_worker.py" --pid-file "$taskWslRoot/$taskOutputRelative/worker-pid.txt" --run-token $taskRunToken --mode $Mode
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
