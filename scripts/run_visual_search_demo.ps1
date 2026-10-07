param([ValidateSet('baseline','oft','teacher')][string]$Model='oft',[string]$Checkpoint='outputs/aerovla_oft/checkpoints/pilot',[ValidateSet('blue_cone','orange_ball')][string]$Target='blue_cone',[string[]]$Cases=@('A','B','C'),[ValidateRange(1,20)][int]$Episodes=1,[int]$SeedStart=0,[string]$Distro='Ubuntu')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskPython=Join-Path $taskRoot 'assets/projectairsim-env/Scripts/python.exe'
$taskOutputRelative='outputs/visual_search/demo'
$taskOutput=Join-Path $taskRoot $taskOutputRelative
if ($Model -eq 'oft' -and -not (Test-Path -LiteralPath (Join-Path $taskRoot "$Checkpoint/manifest.json"))) { throw "No AeroVLA-OFT checkpoint at $Checkpoint; train one first (docs/aerovla_oft.md) or use -Model baseline" }
New-Item -ItemType Directory -Force $taskOutput | Out-Null
Get-ChildItem -LiteralPath $taskOutput -File -ErrorAction SilentlyContinue | Remove-Item -Force
Write-Host "VISUAL SEARCH | model: $Model | target: $Target | the policy sees Front/Down RGB and one sentence only"
$taskViewer=Start-Process -FilePath $taskPython -WorkingDirectory $taskRoot -ArgumentList @((Join-Path $PSScriptRoot 'visual_search_viewer.py'),'--output',$taskOutput) -WindowStyle Hidden -PassThru
try {
    $taskArgs=@{Policy=$Model;Cases=$Cases;Targets=@($Target);Episodes=$Episodes;SeedStart=$SeedStart;Output=$taskOutputRelative;Live=$true;ShowSimulator=$true;Distro=$Distro}
    if ($Model -eq 'oft') { $taskArgs.Checkpoint=$Checkpoint }
    & (Join-Path $PSScriptRoot 'run_visual_search.ps1') @taskArgs
    Write-Host 'Run complete. The observer keeps the last frame; press Q/Esc in it to close.'
    while (-not $taskViewer.HasExited) { Start-Sleep -Milliseconds 300; $taskViewer.Refresh() }
} finally {
    if ($taskViewer -and -not $taskViewer.HasExited) { Stop-Process -Id $taskViewer.Id }
}
