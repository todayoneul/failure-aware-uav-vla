param([ValidateSet('baseline','teacher','oft')][string]$Policy='baseline',[string]$Checkpoint='',[string[]]$Cases=@('A','B','C'),[string[]]$Targets=@('blue_cone','orange_ball'),[ValidateRange(1,200)][int]$Episodes=2,[int]$SeedStart=0,[string]$Output='outputs/visual_search/run',[string]$Record='',[switch]$Resume,[switch]$Live,[string]$Distro='Ubuntu',[switch]$ShowSimulator)
$ErrorActionPreference='Stop'
# `powershell -File` hands a comma list over as one string.
$Cases=@($Cases | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$Targets=@($Targets | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
. (Join-Path $PSScriptRoot 'native_process_args.ps1')
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskExe=Join-Path $taskRoot 'assets/projectairsim-blocks-1.0.1/Blocks/Binaries/Win64/Blocks-Win64-Shipping.exe'
if (-not (Test-Path -LiteralPath $taskExe)) { throw 'Prepared Blocks required; see docs/setup.md' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'outputs/integration/model-downloads.json'))) { throw 'Local model manifest required; see docs/setup.md. This launcher never downloads models.' }
if ($Policy -eq 'oft' -and -not $Checkpoint) { throw 'AeroVLA-OFT needs -Checkpoint <directory>' }
if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in 8989,8990) { throw 'Close the existing simulator/client first.' }
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
    $taskSim=Start-Process -FilePath $taskExe -WorkingDirectory (Split-Path $taskExe) -ArgumentList @('-windowed','-ResX=640','-ResY=480') -WindowStyle $taskWindow -PassThru
    $taskDeadline=(Get-Date).AddSeconds(45)
    do {
        Start-Sleep -Milliseconds 500
        $taskPorts=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $taskSim.Id -and $_.LocalPort -in 8989,8990 })
    } while ($taskPorts.Count -lt 2 -and (Get-Date) -lt $taskDeadline -and -not $taskSim.HasExited)
    if ($taskPorts.Count -lt 2) { throw 'Simulator ports did not become ready' }
    $taskArgs=@('-d',$Distro,'--exec',$taskWslPython,"$taskWslRoot/scripts/visual_search.py",'--host',$taskHost,'--policy',$Policy,'--output',"$taskWslRoot/$($Output.Replace('\','/'))",'--episodes',"$Episodes",'--seed-start',"$SeedStart")
    $taskArgs+='--cases'; $taskArgs+=$Cases
    $taskArgs+='--targets'; $taskArgs+=$Targets
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
        if ($taskRemaining -and $taskRemaining.Path -eq $taskExe) { Stop-Process -Id $taskSim.Id }
    }
}
