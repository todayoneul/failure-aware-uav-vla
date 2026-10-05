param([ValidateRange(1,60)][int]$MaxSteps=30,[switch]$AutoTest,[string]$Distro='Ubuntu')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskOutput=Join-Path $taskRoot 'outputs/failure_demo'
$taskExe=Join-Path $taskRoot 'assets/projectairsim-blocks-1.0.1/Blocks/Binaries/Win64/Blocks-Win64-Shipping.exe'
$taskPython=Join-Path $taskRoot 'assets/projectairsim-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskExe) -or -not (Test-Path -LiteralPath $taskPython)) { throw 'Prepared Blocks and Windows client required; see docs/setup.md' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'outputs/integration/model-downloads.json'))) { throw 'Local model manifest required; see docs/setup.md. This launcher never downloads models.' }
if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in 8989,8990) { throw 'Close the existing simulator/client before starting this demo.' }
$taskWslRoot=& wsl -d $Distro -- wslpath -u $taskRoot.Replace('\','/')
if ($LASTEXITCODE -ne 0 -or -not $taskWslRoot) { throw 'WSL repository path unavailable' }
$taskWslRoot=$taskWslRoot.Trim()
$taskWslHome=(& wsl -d $Distro -- printenv HOME).Trim()
$taskWslPython="$taskWslHome/uav-vla-smoke/integration/bin/python"
& wsl -d $Distro -- test -x $taskWslPython
if ($LASTEXITCODE -ne 0) { throw 'Prepared WSL integration env required; see docs/setup.md' }
$taskHost=((& wsl -d $Distro -- ip -4 route show default) -split '\s+')[2]
if (-not $taskHost) { throw 'Current Windows NAT host address unavailable' }
New-Item -ItemType Directory -Force $taskOutput | Out-Null
if ($AutoTest) { $MaxSteps=6 }
Push-Location $taskRoot
try {
& $taskPython -c 'from src.failures.control import default_control,write_control;write_control("outputs/failure_demo/control.json",default_control())'
if ($LASTEXITCODE -ne 0) { throw 'Control initialization failed; run this script from the repository root' }
& $taskPython -c "from src.integration.blur_demo_support import BlurDemoSession;BlurDemoSession('.', 'outputs/failure_demo', $MaxSteps)"
if ($LASTEXITCODE -ne 0) { throw 'Fresh viewer telemetry initialization failed' }
} finally { Pop-Location }
$taskSim=$taskMonitor=$taskViewer=$taskWorker=$null
try {
    Write-Host 'Gaussian Blur Demo | B: toggle | 1/2/3: severity | Q/Esc: exit'
    Write-Host 'Click the observer window, or use its buttons. Keys apply at the next observation.'
    $taskSim=Start-Process -FilePath $taskExe -WorkingDirectory (Split-Path $taskExe) -ArgumentList @('-windowed','-ResX=1280','-ResY=720','-WinX=0','-WinY=0') -WindowStyle Normal -PassThru
    $taskDeadline=(Get-Date).AddSeconds(35)
    do {
        Start-Sleep -Milliseconds 500
        $taskPorts=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $taskSim.Id -and $_.LocalPort -in 8989,8990 })
    } while ($taskPorts.Count -lt 2 -and (Get-Date) -lt $taskDeadline -and -not $taskSim.HasExited)
    if ($taskPorts.Count -lt 2) { throw 'Simulator ports did not become ready' }
    $taskPS=(Get-Process -Id $PID).Path
    $taskMonitor=Start-Process -FilePath $taskPS -ArgumentList @('-NoProfile','-File',(Join-Path $PSScriptRoot 'communication_final_resources.ps1'),'-SimulatorProcessId',"$($taskSim.Id)",'-OutputDirectory','outputs/failure_demo') -WindowStyle Hidden -PassThru
    $taskViewerArgs=@((Join-Path $PSScriptRoot 'blur_demo_viewer.py'))
    if ($AutoTest) { $taskViewerArgs+='--auto-close' }
    $taskViewer=Start-Process -FilePath $taskPython -WorkingDirectory $taskRoot -ArgumentList $taskViewerArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'viewer.log') -RedirectStandardError (Join-Path $taskOutput 'viewer-errors.log')
    $taskRunnerArgs=@('-d',$Distro,'--',$taskWslPython,"$taskWslRoot/src/integration/closed_loop_runner.py",'--host',$taskHost,'--blur-demo','--steps',"$MaxSteps")
    if ($AutoTest) { $taskRunnerArgs+='--auto-test' }
    $taskWorker=Start-Process -FilePath 'wsl.exe' -ArgumentList $taskRunnerArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'worker.log') -RedirectStandardError (Join-Path $taskOutput 'worker-errors.log')
    Write-Host 'WSL is loading the existing AeroVLA checkpoint. Progress is shown in the observer and worker.log.'
    $taskPrevious=''
    while (-not $taskWorker.HasExited) {
        if ($taskViewer.HasExited) {
            & $taskPython -c 'from src.failures.control import read_control,write_control;p="outputs/failure_demo/control.json";s=read_control(p);s["quit"]=True;write_control(p,s)'
        }
        try {
            $taskStatus=Get-Content -LiteralPath (Join-Path $taskOutput 'telemetry.json') -Raw | ConvertFrom-Json
            $taskProgress="Step $($taskStatus.step)/$MaxSteps | $($taskStatus.phase)"
            if ($taskProgress -ne $taskPrevious) { Write-Host $taskProgress; $taskPrevious=$taskProgress }
        } catch {}
        Start-Sleep -Milliseconds 200
        $taskWorker.Refresh(); $taskViewer.Refresh()
    }
    if ($taskWorker.ExitCode -ne 0) { throw 'WSL demo failed; inspect outputs/failure_demo/worker-errors.log and closed-loop.json' }
    if ($AutoTest) { $null=$taskViewer.WaitForExit(10000) }
    else {
        Write-Host 'Run complete. Inspect the last input/action in the observer; press Q/Esc to close.'
        while (-not $taskViewer.HasExited) { Start-Sleep -Milliseconds 200; $taskViewer.Refresh() }
    }
} finally {
    if ($taskWorker -and -not $taskWorker.HasExited) {
        & $taskPython -c 'from src.failures.control import read_control,write_control;p="outputs/failure_demo/control.json";s=read_control(p);s["quit"]=True;write_control(p,s)'
        $null=$taskWorker.WaitForExit(45000)
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
}
