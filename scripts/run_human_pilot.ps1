param([string]$Scene='field',[string]$Layout='a',[string]$Target='',[ValidateSet('','land','approach')][string]$Task='',[switch]$NoRecord,[string]$Record='',[switch]$ShowSimulator,[string]$Script='',[int]$Episodes=0,[int]$Seed=-1,[int]$TopicsPort=8989,[int]$ServicesPort=8990,[string]$Snapshot='')
# Human pilot mode (docs/human_pilot_mode.md): a person flies the vehicle from the front and down pictures alone, and the
# flight is recorded as data in the format of the teacher's recordings. No model is loaded; nothing runs in WSL.
# The simulator's own window is hidden (it shows the vehicle from outside, which a policy never sees); -ShowSimulator shows
# it, for checking the setup only. -NoRecord flies without writing anything. -Script FILE flies a fixed list of inputs
# instead of a person (a check of the whole loop).
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskExe=Join-Path $taskRoot 'assets/projectairsim-blocks-1.0.1/Blocks/Binaries/Win64/Blocks-Win64-Shipping.exe'
$taskPython=Join-Path $taskRoot 'assets/projectairsim-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskExe) -or -not (Test-Path -LiteralPath $taskPython)) { throw 'Prepared Blocks and Windows client required; see docs/setup.md' }
if (Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in $TopicsPort,$ServicesPort) { throw 'Close the existing simulator/client first, or give other ports (-TopicsPort, -ServicesPort).' }
$taskArgs=@((Join-Path $PSScriptRoot 'human_pilot.py'),'--host','127.0.0.1','--ports',"$TopicsPort","$ServicesPort",'--scene',$Scene,'--layout',$Layout)
if ($Target) { $taskArgs+='--target'; $taskArgs+=$Target }
if ($Task) { $taskArgs+='--task'; $taskArgs+=$Task }
if ($NoRecord) { $taskArgs+='--no-record' }
if ($Record) { $taskArgs+='--record'; $taskArgs+=$Record }
if ($Script) { $taskArgs+='--script'; $taskArgs+=$Script }
if ($Episodes -gt 0) { $taskArgs+='--episodes'; $taskArgs+="$Episodes" }
if ($Seed -ge 0) { $taskArgs+='--seed'; $taskArgs+="$Seed" }
if ($Snapshot) { $taskArgs+='--snapshot'; $taskArgs+=$Snapshot }
Write-Host 'HUMAN PILOT'
Write-Host 'Before a flight: N/P object | T land/approach | L layout | M scene | Enter fly | Esc quit'
Write-Host 'Flying: W/Up forward | A D turn | R/PgUp up | F/PgDn down | 1 2 3 speed | Enter done | Backspace discard'
Write-Host 'Pad (Xbox-style): left stick forward and turn | right stick up/down | A done | B discard | Start fly | LB/RB speed'
Write-Host 'The window shows only the front and down pictures a policy is given, and the sentence.'
$taskSim=$null
try {
    $taskWindow=if($ShowSimulator){'Normal'}else{'Hidden'}
    $taskSimArgs=@('-windowed','-ResX=640','-ResY=480')
    if ($TopicsPort -ne 8989 -or $ServicesPort -ne 8990) { $taskSimArgs+="-topicsport=$TopicsPort"; $taskSimArgs+="-servicesport=$ServicesPort" }
    $taskSim=Start-Process -FilePath $taskExe -WorkingDirectory (Split-Path $taskExe) -ArgumentList $taskSimArgs -WindowStyle $taskWindow -PassThru
    $taskDeadline=(Get-Date).AddSeconds(45)
    do {
        Start-Sleep -Milliseconds 500
        $taskPorts=@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $taskSim.Id -and $_.LocalPort -in $TopicsPort,$ServicesPort })
    } while ($taskPorts.Count -lt 2 -and (Get-Date) -lt $taskDeadline -and -not $taskSim.HasExited)
    if ($taskPorts.Count -lt 2) { throw 'Simulator ports did not become ready' }
    # The simulator client logs to stderr; Windows PowerShell would stop on the first such line under 'Stop'.
    $ErrorActionPreference='Continue'
    & $taskPython @taskArgs
    $ErrorActionPreference='Stop'
    if ($LASTEXITCODE -ne 0) { throw "Human pilot ended with exit $LASTEXITCODE; see outputs/human_pilot" }
} finally {
    if ($taskSim) {
        $taskRemaining=Get-Process -Id $taskSim.Id -ErrorAction SilentlyContinue
        if ($taskRemaining -and $taskRemaining.Path -eq $taskExe) { Stop-Process -Id $taskSim.Id; $null=$taskRemaining.WaitForExit(15000) }
    }
}
