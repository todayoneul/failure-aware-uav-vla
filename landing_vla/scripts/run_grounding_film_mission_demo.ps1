param([string]$Checkpoint='outputs/aerovla_oft/checkpoints/grounding_film',[ValidateRange(0,400)][int]$MaxSteps=0,[string]$Distro='Ubuntu',[Nullable[double]]$StartX=$null,[Nullable[double]]$StartY=$null,[Nullable[double]]$StartYaw=$null,[Nullable[double]]$StartHeight=$null,[string]$Layout='')
# Interactive Mission Control flown by the frozen AeroVLA-OFT baseline (docs/grounding_film_interactive_demo.md).
# The policy is given Front RGB, Down RGB and one sentence; the map and the target's place are for the viewer and the evaluator.
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
# Decisions per mission: the evaluation's own budget unless one is given.
if (-not $MaxSteps) { $MaxSteps=[int](Get-Content -LiteralPath (Join-Path $taskRoot 'configs/visual_search.json') -Raw | ConvertFrom-Json).gen_v3.max_ticks }
# A start of the user's own instead of the first preset launch, and another layout of the map, when given
# (docs/arbitrary_start_generalized_landing.md): .\scripts\run_grounding_film_mission_demo.ps1 -StartX 30 -StartY 40 -StartYaw 135 -StartHeight 8
$taskStart=@{}
foreach ($taskName in @('StartX','StartY','StartYaw','StartHeight')) { if ($null -ne (Get-Variable $taskName -ValueOnly)) { $taskStart[$taskName]=(Get-Variable $taskName -ValueOnly) } }
if ($Layout) { $taskStart.Layout=$Layout }
& (Join-Path $PSScriptRoot 'run_blur_demo.ps1') -Mode mission -Policy grounding-film -Checkpoint $Checkpoint -MaxSteps $MaxSteps -Distro $Distro @taskStart
