param([int]$SimulatorProcessId,[int]$DurationSeconds=1800,[string]$OutputDirectory='outputs/communication_final')
$ErrorActionPreference='Stop'
$taskOutput=Join-Path (Split-Path -Parent $PSScriptRoot) $OutputDirectory
New-Item -ItemType Directory -Force $taskOutput | Out-Null
$taskDeadline=(Get-Date).AddSeconds($DurationSeconds)
while ((Get-Date) -lt $taskDeadline -and (Get-Process -Id $SimulatorProcessId -ErrorAction SilentlyContinue)) {
    $taskOS=Get-CimInstance Win32_OperatingSystem
    $taskMem=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $taskPages=@(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage)
    $taskGPU=& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    $taskPIDMemory=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object Name -match "pid_${SimulatorProcessId}_" | Select-Object Name,DedicatedUsage,SharedUsage)
    $taskStage=if(Test-Path (Join-Path $taskOutput 'active-stage.txt')) {(Get-Content -LiteralPath (Join-Path $taskOutput 'active-stage.txt') -Raw).Trim()} else {'initial'}
    $taskRow=[ordered]@{epoch=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000;stage=$taskStage;simulator_pid=$SimulatorProcessId;windows_used_GiB=($taskOS.TotalVisibleMemorySize-$taskOS.FreePhysicalMemory)/1MB;windows_available_GiB=$taskOS.FreePhysicalMemory/1MB;commit_GiB=$taskMem.CommittedBytes/1GB;commit_limit_GiB=$taskMem.CommitLimit/1GB;pagefiles=$taskPages;global_nvidia_smi=$taskGPU;simulator_gpu_memory=$taskPIDMemory}
    $taskText=$taskRow | ConvertTo-Json -Depth 6 -Compress
    $taskText | Add-Content -LiteralPath (Join-Path $taskOutput 'windows-resources.jsonl') -Encoding utf8
    $taskText | Set-Content -LiteralPath (Join-Path $taskOutput 'windows-latest.tmp') -Encoding utf8
    Move-Item -LiteralPath (Join-Path $taskOutput 'windows-latest.tmp') -Destination (Join-Path $taskOutput 'windows-latest.json') -Force
    Start-Sleep -Seconds 1
}
