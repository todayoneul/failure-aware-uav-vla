param([string]$Checkpoint='outputs/aerovla_oft/checkpoints/grounding_film',[ValidateRange(0,400)][int]$MaxSteps=0,[string]$Distro='Ubuntu')
# Interactive Mission Control flown by the frozen AeroVLA-OFT baseline (docs/grounding_film_interactive_demo.md).
# The policy is given Front RGB, Down RGB and one sentence; the map and the target's place are for the viewer and the evaluator.
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
# Decisions per mission: the evaluation's own budget unless one is given.
if (-not $MaxSteps) { $MaxSteps=[int](Get-Content -LiteralPath (Join-Path $taskRoot 'configs/visual_search.json') -Raw | ConvertFrom-Json).gen_v3.max_ticks }
& (Join-Path $PSScriptRoot 'run_blur_demo.ps1') -Mode mission -Policy grounding-film -Checkpoint $Checkpoint -MaxSteps $MaxSteps -Distro $Distro
