param([int]$SimulatorProcessId, [int]$DurationSeconds=100)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOutput = Join-Path $taskRoot 'outputs\platform_final'
$samplePath = Join-Path $taskOutput 'wddm-samples.jsonl'
$deadline = (Get-Date).AddSeconds($DurationSeconds)
(Get-Process -Id $SimulatorProcessId).Modules | Where-Object ModuleName -match 'd3d|dxgi|nvwg|vulkan|llvmpipe' | Select-Object ModuleName,FileName | ConvertTo-Json | Set-Content (Join-Path $taskOutput 'renderer-modules.json')
while ((Get-Date) -lt $deadline -and (Get-Process -Id $SimulatorProcessId -ErrorAction SilentlyContinue)) {
    $memory = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object Name -match "pid_${SimulatorProcessId}_" | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted)
    $engines = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | Where-Object Name -match "pid_${SimulatorProcessId}_" | Select-Object Name,UtilizationPercentage)
    $gpu = & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    [ordered]@{epoch_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds();memory=$memory;engines=$engines;global_nvidia_smi=$gpu} | ConvertTo-Json -Depth 5 -Compress | Add-Content -LiteralPath $samplePath -Encoding utf8
    Start-Sleep -Seconds 1
}
