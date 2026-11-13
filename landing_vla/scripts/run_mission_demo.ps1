param([ValidateRange(0,400)][int]$MaxSteps=0,[string]$Distro='Ubuntu',[ValidateSet('step','continuous')][string]$Flight='continuous',[ValidateSet('legacy','grounding-film')][string]$Policy='legacy',[string]$Checkpoint='',[Nullable[double]]$StartX=$null,[Nullable[double]]$StartY=$null,[Nullable[double]]$StartYaw=$null,[Nullable[double]]$StartHeight=$null,[string]$Layout='')
$ErrorActionPreference='Stop'
if ($Policy -eq 'grounding-film') {
    # The same Mission Control flown by an AeroVLA-OFT checkpoint; -Flight does not apply (it flies as the evaluation does).
    $taskArgs=@{MaxSteps=$MaxSteps;Distro=$Distro}
    if ($Checkpoint) { $taskArgs.Checkpoint=$Checkpoint }
    foreach ($taskName in @('StartX','StartY','StartYaw','StartHeight')) { if ($null -ne (Get-Variable $taskName -ValueOnly)) { $taskArgs[$taskName]=(Get-Variable $taskName -ValueOnly) } }
    if ($Layout) { $taskArgs.Layout=$Layout }
    & (Join-Path $PSScriptRoot 'run_grounding_film_mission_demo.ps1') @taskArgs
    return
}
if ($Checkpoint) { throw '-Checkpoint is for -Policy grounding-film' }
if ($null -ne $StartX -or $null -ne $StartY -or $null -ne $StartYaw -or $null -ne $StartHeight -or $Layout) { throw '-StartX, -StartY, -StartYaw, -StartHeight and -Layout are for -Policy grounding-film' }
if (-not $MaxSteps) { $MaxSteps=60 }
if ($MaxSteps -gt 60) { throw 'MaxSteps within 1..60 for the legacy policy' }
& (Join-Path $PSScriptRoot 'run_blur_demo.ps1') -Mode mission -MaxSteps $MaxSteps -Distro $Distro -Flight $Flight
