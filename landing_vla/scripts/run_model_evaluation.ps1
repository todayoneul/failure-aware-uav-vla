param([string[]]$Only=@(),[string]$Output='outputs/model_eval/run',[switch]$Resume,[string]$Suffix='',[ValidateSet('','step','continuous')][string]$Flight='',[ValidateRange(0,60)][int]$MaxSteps=0,[ValidateRange(1,20)][int]$Repeats=1,[ValidateSet('','grammar','free')][string]$Decoder='',[string]$Protocol='',[string]$Distro='Ubuntu',[switch]$ShowSimulator)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native_process_args.ps1')
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskExe=Join-Path $taskRoot 'assets/projectairsim-blocks-1.0.1/Blocks/Binaries/Win64/Blocks-Win64-Shipping.exe'
if (-not (Test-Path -LiteralPath $taskExe)) { throw 'Prepared Blocks required; see docs/setup.md' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'outputs/integration/model-downloads.json'))) { throw 'Local model manifest required; see docs/setup.md. This launcher never downloads models.' }
if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in 8989,8990) { throw 'Close the existing simulator/client before starting the evaluation.' }
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
    $taskArgs=@('-d',$Distro,'--exec',$taskWslPython,"$taskWslRoot/scripts/evaluate_model.py",'--host',$taskHost,'--output',"$taskWslRoot/$Output")
    if ($Resume) { $taskArgs+='--resume' }
    if ($Suffix) { $taskArgs+='--suffix'; $taskArgs+=$Suffix }
    if ($Flight) { $taskArgs+='--flight'; $taskArgs+=$Flight }
    if ($MaxSteps) { $taskArgs+='--max-steps'; $taskArgs+="$MaxSteps" }
    if ($Repeats -gt 1) { $taskArgs+='--repeats'; $taskArgs+="$Repeats" }
    if ($Decoder) { $taskArgs+='--decoder'; $taskArgs+=$Decoder }
    if ($Protocol) { $taskArgs+='--protocol'; $taskArgs+="$taskWslRoot/$($Protocol.Replace('\','/'))" }
    if ($Only.Count) { $taskArgs+='--only'; $taskArgs+=$Only }
    Write-Host "Model evaluation running; progress: $Output/worker.log"
    $taskWorker=Start-Process -FilePath 'wsl.exe' -ArgumentList (Join-NativeArguments $taskArgs) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'worker.log') -RedirectStandardError (Join-Path $taskOutput 'worker-errors.log')
    Enable-ProcessExitTracking $taskWorker
    $taskWorker.WaitForExit()
    if ($taskWorker.ExitCode -ne 0) { throw "Evaluation worker failed (exit $($taskWorker.ExitCode)); inspect $Output/worker.log and worker-errors.log" }
    Write-Host "Evaluation complete: $Output/results.json"
} finally {
    if ($taskSim) {
        $taskRemaining=Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue
        if ($taskRemaining -and $taskRemaining.Path -eq $taskExe) { Stop-Process -Id $taskSim.Id }
    }
}
