param([ValidateRange(1,60)][int]$MaxSteps=60,[string]$Distro='Ubuntu')
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'run_blur_demo.ps1') -Mode mission -MaxSteps $MaxSteps -Distro $Distro
