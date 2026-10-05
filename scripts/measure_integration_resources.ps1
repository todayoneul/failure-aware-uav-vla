param([int]$SimulatorProcessId=0,[int]$DurationSeconds=600,[string]$Stage='model')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskOutput=Join-Path $taskRoot 'outputs/integration'
$taskPath=Join-Path $taskOutput "$Stage-windows-resources.jsonl"
$taskDeadline=(Get-Date).AddSeconds($DurationSeconds)
while ((Get-Date) -lt $taskDeadline) {
    if ($SimulatorProcessId -gt 0 -and -not (Get-Process -Id $SimulatorProcessId -ErrorAction SilentlyContinue)) {break}
    $taskOS=Get-CimInstance Win32_OperatingSystem
    $taskMemory=@();$taskEngines=@()
    if ($SimulatorProcessId -gt 0) {
        $taskMemory=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object Name -match "pid_${SimulatorProcessId}_" | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted)
        $taskEngines=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | Where-Object Name -match "pid_${SimulatorProcessId}_" | Select-Object Name,UtilizationPercentage)
    }
    $taskGPU=& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    [ordered]@{epoch_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds();stage=$Stage;simulator_pid=$SimulatorProcessId;windows_system_used_GiB=($taskOS.TotalVisibleMemorySize-$taskOS.FreePhysicalMemory)/1MB;simulator_memory=$taskMemory;simulator_engines=$taskEngines;global_nvidia_smi=$taskGPU} | ConvertTo-Json -Depth 5 -Compress | Add-Content -LiteralPath $taskPath -Encoding utf8
    Start-Sleep -Seconds 1
}
