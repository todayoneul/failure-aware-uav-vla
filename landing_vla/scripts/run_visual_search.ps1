param([ValidateSet('baseline','teacher','oft')][string]$Policy='baseline',[string]$Checkpoint='',[string[]]$Cases=@('A','B','C'),[string[]]$Targets=@('blue_cone','orange_ball'),[ValidateRange(1,200)][int]$Episodes=2,[int]$SeedStart=0,[string]$Output='outputs/visual_search/run',[string]$Record='',[switch]$Resume,[switch]$Live,[string]$Distro='Ubuntu',[switch]$ShowSimulator,[string]$Plan='',[string]$Set='',[string[]]$Only=@(),[int]$Skip=0,[int]$Limit=0,[string]$Layout='',[string]$Strategy='',[ValidateSet('step','continuous')][string]$Flight='step',[string]$ModelName='',[int]$TopicsPort=8989,[int]$ServicesPort=8990,[switch]$PilotVerbs,[string]$Named='',[ValidateSet('on','off')][string]$Finalizer='on')
$ErrorActionPreference='Stop'
# `powershell -File` hands a comma list over as one string.
$Cases=@($Cases | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$Targets=@($Targets | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$Only=@($Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
if ($Plan -and -not $Set) { throw '-Plan needs -Set (G1, G2, G3, G4, P, S, or a split of a dataset plan)' }
. (Join-Path $PSScriptRoot 'native_process_args.ps1')
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskExe=Join-Path $taskRoot 'assets/projectairsim-blocks-1.0.1/Blocks/Binaries/Win64/Blocks-Win64-Shipping.exe'
if (-not (Test-Path -LiteralPath $taskExe)) { throw 'Prepared Blocks required; see docs/setup.md' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'outputs/integration/model-downloads.json'))) { throw 'Local model manifest required; see docs/setup.md. This launcher never downloads models.' }
if ($Policy -eq 'oft' -and -not $Checkpoint) { throw 'AeroVLA-OFT needs -Checkpoint <directory>' }
# A simulator that was just closed by a previous run needs a moment to release its ports.
$taskPortDeadline=(Get-Date).AddSeconds(30)
while ((Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in $TopicsPort,$ServicesPort) -and (Get-Date) -lt $taskPortDeadline) { Start-Sleep -Milliseconds 500 }
if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in $TopicsPort,$ServicesPort) { throw 'Close the existing simulator/client first.' }
$taskWslRoot=(& wsl -d $Distro --exec wslpath -u $taskRoot.Replace('\','/')).Trim()
$taskWslHome=(& wsl -d $Distro --exec printenv HOME).Trim()
$taskWslPython="$taskWslHome/uav-vla-smoke/integration/bin/python"
& wsl -d $Distro --exec test -x $taskWslPython
if ($LASTEXITCODE -ne 0) { throw 'Prepared WSL integration env required; see docs/setup.md' }
$taskHost=((& wsl -d $Distro --exec ip -4 route show default) -split '\s+')[2]
if (-not $taskHost) { throw 'Current Windows NAT host address unavailable' }
$taskOutput=Join-Path $taskRoot $Output
New-Item -ItemType Directory -Force $taskOutput | Out-Null
$taskSim=$null
try {
    $taskWindow=if($ShowSimulator){'Normal'}else{'Hidden'}
    $taskSimArgs=@('-windowed','-ResX=640','-ResY=480')
    # A second simulator for a parallel run listens on its own pair of ports.
    if ($TopicsPort -ne 8989 -or $ServicesPort -ne 8990) { $taskSimArgs+="-topicsport=$TopicsPort"; $taskSimArgs+="-servicesport=$ServicesPort" }
    $taskSim=Start-Process -FilePath $taskExe -WorkingDirectory (Split-Path $taskExe) -ArgumentList $taskSimArgs -WindowStyle $taskWindow -PassThru
    $taskDeadline=(Get-Date).AddSeconds(45)
    do {
        Start-Sleep -Milliseconds 500
        $taskPorts=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $taskSim.Id -and $_.LocalPort -in $TopicsPort,$ServicesPort })
    } while ($taskPorts.Count -lt 2 -and (Get-Date) -lt $taskDeadline -and -not $taskSim.HasExited)
    if ($taskPorts.Count -lt 2) { throw 'Simulator ports did not become ready' }
    $taskArgs=@('-d',$Distro,'--exec',$taskWslPython,"$taskWslRoot/scripts/visual_search.py",'--host',$taskHost,'--policy',$Policy,'--output',"$taskWslRoot/$($Output.Replace('\','/'))",'--episodes',"$Episodes",'--seed-start',"$SeedStart",'--flight',$Flight,'--ports',"$TopicsPort","$ServicesPort")
    if ($Plan) {
        $taskArgs+='--plan'; $taskArgs+="$taskWslRoot/$($Plan.Replace('\','/'))"; $taskArgs+='--set'; $taskArgs+=$Set
        if ($Only.Count) { $taskArgs+='--only'; $taskArgs+=$Only }
        if ($Skip) { $taskArgs+='--skip'; $taskArgs+="$Skip" }
        if ($Limit) { $taskArgs+='--limit'; $taskArgs+="$Limit" }
        if ($Layout) { $taskArgs+='--layout'; $taskArgs+=$Layout }
        if ($Strategy) { $taskArgs+='--strategy'; $taskArgs+=$Strategy }
        if ($PilotVerbs) { $taskArgs+='--pilot-verbs' }
        if ($Named) { $taskArgs+='--named'; $taskArgs+=$Named }
    } else {
        $taskArgs+='--cases'; $taskArgs+=$Cases
        $taskArgs+='--targets'; $taskArgs+=$Targets
    }
    if ($ModelName) { $taskArgs+='--model-name'; $taskArgs+=$ModelName }
    # The low-level landing finalizer is on unless a run asks to fly as before it existed.
    if ($Finalizer -eq 'off') { $taskArgs+='--finalizer'; $taskArgs+='off' }
    if ($Checkpoint) { $taskArgs+='--checkpoint'; $taskArgs+="$taskWslRoot/$($Checkpoint.Replace('\','/'))" }
    if ($Record) { $taskArgs+='--record'; $taskArgs+="$taskWslRoot/$($Record.Replace('\','/'))" }
    if ($Resume) { $taskArgs+='--resume' }
    if ($Live) { $taskArgs+='--live' }
    Write-Host "Visual search ($Policy) running; progress: $Output/worker.log"
    $taskWorker=Start-Process -FilePath 'wsl.exe' -ArgumentList (Join-NativeArguments $taskArgs) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'worker.log') -RedirectStandardError (Join-Path $taskOutput 'worker-errors.log')
    Enable-ProcessExitTracking $taskWorker
    $taskWorker.WaitForExit()
    if ($taskWorker.ExitCode -ne 0) { throw "Visual search worker failed (exit $($taskWorker.ExitCode)); inspect $Output/worker.log and worker-errors.log" }
    Write-Host "Visual search complete: $Output/results.json"
} finally {
    if ($taskSim) {
        $taskRemaining=Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue
        if ($taskRemaining -and $taskRemaining.Path -eq $taskExe) { Stop-Process -Id $taskSim.Id; $null=$taskRemaining.WaitForExit(15000) }
    }
}
