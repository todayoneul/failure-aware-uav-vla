param([ValidateSet('main','repeat')][string]$Phase='main',[int]$Limit=0,[string]$Output='outputs/failures/gaussian_blur',[string]$Starts='',[ValidateRange(1,20)][int]$Attempts=8,[string]$Distro='Ubuntu',[switch]$HideSimulator,[int]$TopicsPort=8989,[int]$ServicesPort=8990)
# Gaussian blur robustness characterization (docs/failure_gaussian_blur.md): the canonical test starts, clean and at three
# blur severities, flown by the frozen baseline. The simulator is started as scripts/run_visual_search.ps1 starts it for an
# evaluation (same arguments), with its window shown: a hidden simulator stops advancing when the desktop session is
# disconnected (configs/failures/gaussian_blur_characterization.json, runtime). The worker is scripts/blur_characterization.py.
# The frozen baseline's fingerprints are verified before the simulator starts and again after the worker ends. A worker
# that stops with flights left (a harness error, a simulator that no longer answers) is started again with a new simulator;
# what was flown is kept and never flown again.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native_process_args.ps1')
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskExe=Join-Path $taskRoot 'assets/projectairsim-blocks-1.0.1/Blocks/Binaries/Win64/Blocks-Win64-Shipping.exe'
$taskPython=Join-Path $taskRoot 'assets/projectairsim-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskExe) -or -not (Test-Path -LiteralPath $taskPython)) { throw 'Prepared Blocks and Windows client required; see docs/setup.md' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'outputs/integration/model-downloads.json'))) { throw 'Local model manifest required; see docs/setup.md. This launcher never downloads models.' }
$taskWslRoot=(& wsl -d $Distro --exec wslpath -u $taskRoot.Replace('\','/')).Trim()
$taskWslHome=(& wsl -d $Distro --exec printenv HOME).Trim()
$taskWslPython="$taskWslHome/uav-vla-smoke/integration/bin/python"
& wsl -d $Distro --exec test -x $taskWslPython
if ($LASTEXITCODE -ne 0) { throw 'Prepared WSL integration env required; see docs/setup.md' }
$taskOutput=Join-Path $taskRoot $Output
$taskRecords=Join-Path $taskOutput 'config'
New-Item -ItemType Directory -Force $taskRecords | Out-Null
$taskFrozen=(Get-Content -LiteralPath (Join-Path $taskRoot 'configs/failures/gaussian_blur_characterization.json') -Raw | ConvertFrom-Json).frozen_record

function Confirm-FrozenBaseline([string]$When) {
    # Recomputes every fingerprint of the frozen record; a difference ends the experiment.
    $taskResult=& $taskPython (Join-Path $PSScriptRoot 'freeze_baseline.py') verify $taskFrozen
    $taskPassed=($LASTEXITCODE -eq 0)
    $taskLine=@{when=$When;phase=$Phase;utc=(Get-Date).ToUniversalTime().ToString('o');passed=$taskPassed;result=(($taskResult -join '') | ConvertFrom-Json)} | ConvertTo-Json -Compress -Depth 5
    Add-Content -LiteralPath (Join-Path $taskRecords 'frozen_verification.jsonl') -Value $taskLine -Encoding utf8
    if (-not $taskPassed) { throw "The frozen baseline does not match its record ($When); flights made with it are not valid." }
    Write-Host "Frozen baseline verified ($When)."
}

for ($taskAttempt=1; $taskAttempt -le $Attempts; $taskAttempt++) {
    Confirm-FrozenBaseline "before launch $taskAttempt"
    # A simulator that was just closed by a previous run needs a moment to release its ports.
    $taskPortDeadline=(Get-Date).AddSeconds(30)
    while ((Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in $TopicsPort,$ServicesPort) -and (Get-Date) -lt $taskPortDeadline) { Start-Sleep -Milliseconds 500 }
    if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in $TopicsPort,$ServicesPort) { throw 'Close the existing simulator/client first.' }
    $taskHost=((& wsl -d $Distro --exec ip -4 route show default) -split '\s+')[2]
    if (-not $taskHost) { throw 'Current Windows NAT host address unavailable' }
    $taskSim=$null; $taskCode=1
    try {
        $taskWindow=if($HideSimulator){'Hidden'}else{'Normal'}
        $taskSimArgs=@('-windowed','-ResX=640','-ResY=480')
        if ($TopicsPort -ne 8989 -or $ServicesPort -ne 8990) { $taskSimArgs+="-topicsport=$TopicsPort"; $taskSimArgs+="-servicesport=$ServicesPort" }
        $taskSim=Start-Process -FilePath $taskExe -WorkingDirectory (Split-Path $taskExe) -ArgumentList $taskSimArgs -WindowStyle $taskWindow -PassThru
        $taskDeadline=(Get-Date).AddSeconds(45)
        do {
            Start-Sleep -Milliseconds 500
            $taskPorts=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $taskSim.Id -and $_.LocalPort -in $TopicsPort,$ServicesPort })
        } while ($taskPorts.Count -lt 2 -and (Get-Date) -lt $taskDeadline -and -not $taskSim.HasExited)
        if ($taskPorts.Count -lt 2) { throw 'Simulator ports did not become ready' }
        $taskArgs=@('-d',$Distro,'--exec',$taskWslPython,"$taskWslRoot/scripts/blur_characterization.py",'fly','--host',$taskHost,'--ports',"$TopicsPort","$ServicesPort",'--phase',$Phase,'--output',"$taskWslRoot/$($Output.Replace('\','/'))")
        if ($Limit) { $taskArgs+='--limit'; $taskArgs+="$Limit" }
        if ($Starts) { $taskArgs+='--starts'; $taskArgs+=$Starts.Replace('\','/') }
        Write-Host "Blur characterization ($Phase), launch $taskAttempt of at most $Attempts; progress: $Output/worker-$Phase.log"
        $taskWorker=Start-Process -FilePath 'wsl.exe' -ArgumentList (Join-NativeArguments $taskArgs) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput "worker-$Phase-$taskAttempt.log") -RedirectStandardError (Join-Path $taskOutput "worker-$Phase-$taskAttempt-errors.log")
        Enable-ProcessExitTracking $taskWorker
        $taskWorker.WaitForExit()
        $taskCode=$taskWorker.ExitCode
    } finally {
        if ($taskSim) {
            $taskRemaining=Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue
            if ($taskRemaining -and $taskRemaining.Path -eq $taskExe) { Stop-Process -Id $taskSim.Id; $null=$taskRemaining.WaitForExit(15000) }
        }
    }
    Confirm-FrozenBaseline "after launch $taskAttempt"
    if ($taskCode -eq 0) { Write-Host "Blur characterization ($Phase) complete: $Output"; exit 0 }
    if ($taskCode -notin 3,4) { throw "Blur characterization worker failed (exit $taskCode); inspect $Output/worker-$Phase-$taskAttempt.log and its errors log" }
    Write-Host "Worker ended with flights left (exit $taskCode); starting again."
}
throw "Flights remain after $Attempts launches; inspect $Output/flights.jsonl"
