param([string]$Plan='configs/experiments/controlled_landings.json',[string]$Output='outputs/arbitrary_start/controlled',[ValidateSet('controlled','oft')][string]$Policy='controlled',[string]$Checkpoint='outputs/aerovla_oft/checkpoints/grounding_film',[string[]]$Only=@(),[int]$Limit=0,[switch]$Again,[ValidateRange(1,10)][int]$Attempts=3,[string]$Distro='Ubuntu',[switch]$HideSimulator)
# Flies a plan of starts with landing surfaces read as surfaces (docs/arbitrary_start_generalized_landing.md). The worker is
# scripts/surface_flights.py: the canonical episode loop from each episode's own start, flown by the episode's scripted pilot
# (-Policy controlled: flights whose outcome is known beforehand, for checking the evaluator) or by a checkpoint (-Policy oft).
# The simulator is started as scripts/run_blur_characterization.ps1 starts it, with its window shown. The frozen baselines'
# fingerprints are verified before the simulator starts and after the worker ends: nothing here trains or changes a model.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native_process_args.ps1')
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskExe=Join-Path $taskRoot 'assets/projectairsim-blocks-1.0.1/Blocks/Binaries/Win64/Blocks-Win64-Shipping.exe'
$taskPython=Join-Path $taskRoot 'assets/projectairsim-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskExe) -or -not (Test-Path -LiteralPath $taskPython)) { throw 'Prepared Blocks and Windows client required; see docs/setup.md' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot $Plan))) { throw "No plan file at $Plan" }
if ($Policy -eq 'oft' -and -not (Test-Path -LiteralPath (Join-Path $taskRoot 'outputs/integration/model-downloads.json'))) { throw 'Local model manifest required; see docs/setup.md. This launcher never downloads models.' }
$taskWslRoot=(& wsl -d $Distro --exec wslpath -u $taskRoot.Replace('\','/')).Trim()
$taskWslHome=(& wsl -d $Distro --exec printenv HOME).Trim()
$taskWslPython="$taskWslHome/uav-vla-smoke/integration/bin/python"
& wsl -d $Distro --exec test -x $taskWslPython
if ($LASTEXITCODE -ne 0) { throw 'Prepared WSL integration env required; see docs/setup.md' }
$taskOutput=Join-Path $taskRoot $Output
New-Item -ItemType Directory -Force $taskOutput | Out-Null

function Confirm-FrozenBaselines([string]$When) {
    # Recomputes every fingerprint of both frozen records; a difference ends the run.
    foreach ($taskRecord in @('grounding_film','gen_v2')) {
        $taskResult=& $taskPython (Join-Path $PSScriptRoot 'freeze_baseline.py') verify $taskRecord
        $taskPassed=($LASTEXITCODE -eq 0)
        $taskLine=@{when=$When;record=$taskRecord;utc=(Get-Date).ToUniversalTime().ToString('o');passed=$taskPassed;result=(($taskResult -join '') | ConvertFrom-Json)} | ConvertTo-Json -Compress -Depth 5
        # Appended as plain UTF-8 (Windows PowerShell's own utf8 writes a byte-order mark that JSON readers trip over).
        [IO.File]::AppendAllText((Join-Path $taskOutput 'frozen_verification.jsonl'),$taskLine+"`n",(New-Object Text.UTF8Encoding $false))
        if (-not $taskPassed) { throw "The frozen record $taskRecord does not match its fingerprints ($When)." }
    }
    Write-Host "Frozen baselines verified ($When)."
}

for ($taskAttempt=1; $taskAttempt -le $Attempts; $taskAttempt++) {
    Confirm-FrozenBaselines "before launch $taskAttempt"
    $taskPortDeadline=(Get-Date).AddSeconds(30)
    while ((Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in 8989,8990) -and (Get-Date) -lt $taskPortDeadline) { Start-Sleep -Milliseconds 500 }
    if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in 8989,8990) { throw 'Close the existing simulator/client first.' }
    $taskHost=((& wsl -d $Distro --exec ip -4 route show default) -split '\s+')[2]
    if (-not $taskHost) { throw 'Current Windows NAT host address unavailable' }
    $taskSim=$null; $taskCode=1
    try {
        $taskWindow=if($HideSimulator){'Hidden'}else{'Normal'}
        $taskSim=Start-Process -FilePath $taskExe -WorkingDirectory (Split-Path $taskExe) -ArgumentList @('-windowed','-ResX=640','-ResY=480') -WindowStyle $taskWindow -PassThru
        $taskDeadline=(Get-Date).AddSeconds(45)
        do {
            Start-Sleep -Milliseconds 500
            $taskPorts=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $taskSim.Id -and $_.LocalPort -in 8989,8990 })
        } while ($taskPorts.Count -lt 2 -and (Get-Date) -lt $taskDeadline -and -not $taskSim.HasExited)
        if ($taskPorts.Count -lt 2) { throw 'Simulator ports did not become ready' }
        $taskArgs=@('-d',$Distro,'--exec',$taskWslPython,"$taskWslRoot/scripts/surface_flights.py",'--host',$taskHost,'--plan',"$taskWslRoot/$($Plan.Replace('\','/'))",'--output',"$taskWslRoot/$($Output.Replace('\','/'))",'--policy',$Policy)
        if ($Policy -eq 'oft') { $taskArgs+='--checkpoint'; $taskArgs+="$taskWslRoot/$($Checkpoint.Replace('\','/'))" }
        if ($Only.Count) { $taskArgs+='--only'; $taskArgs+=$Only }
        if ($Limit) { $taskArgs+='--limit'; $taskArgs+="$Limit" }
        if ($Again -and $taskAttempt -eq 1) { $taskArgs+='--again' }
        $taskLog="worker-$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))"
        Write-Host "Surface flights ($Policy), launch $taskAttempt of at most $Attempts; progress: $Output/$taskLog.log"
        $taskWorker=Start-Process -FilePath 'wsl.exe' -ArgumentList (Join-NativeArguments $taskArgs) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput "$taskLog.log") -RedirectStandardError (Join-Path $taskOutput "$taskLog-errors.log")
        Enable-ProcessExitTracking $taskWorker
        $taskWorker.WaitForExit()
        $taskCode=$taskWorker.ExitCode
    } finally {
        if ($taskSim) {
            $taskRemaining=Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue
            if ($taskRemaining -and $taskRemaining.Path -eq $taskExe) { Stop-Process -Id $taskSim.Id; $null=$taskRemaining.WaitForExit(15000) }
        }
    }
    Confirm-FrozenBaselines "after launch $taskAttempt"
    if ($taskCode -eq 0) { Write-Host "Surface flights complete, every check passed: $Output"; exit 0 }
    if ($taskCode -eq 1) { Write-Host "Surface flights complete with a failed check or a flight that could not be flown; see $Output/results.json and $taskLog.log"; exit 1 }
    if ($taskCode -ne 4) { throw "Surface flight worker failed (exit $taskCode); inspect $Output/$taskLog.log and its errors log" }
    Write-Host 'The simulator stopped answering; starting again with what is flown kept.'
}
throw "Flights remain after $Attempts launches; inspect $Output"
