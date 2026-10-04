param([switch]$UseReverseBridge)
$ErrorActionPreference = 'Stop'
$TaskRoot = 'C:\Users\leegy\Desktop\drone'
$EvidenceRoot = Join-Path $TaskRoot 'outputs\compatibility'
$SimExe = Join-Path $TaskRoot 'assets\windows-blocks-1.8.1\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe'
$SimArguments = @('-RenderOffscreen', '-d3d11', '-NoSound', '-NoVSync', '-ResX=640', '-ResY=480', '-windowed', "-settings=$TaskRoot\scripts\gate2-settings.json", "-AbsLog=$EvidenceRoot\windows-blocks-unreal.log")
$Samples = [System.Collections.Generic.List[object]]::new()
$GpuBaseline = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim()
$SimProcess = $null
$ProbeExit = $null
$WslBridgeProcess = $null
$WindowsBridgeProcess = $null
$RunLabel = if ($UseReverseBridge) { 'windows-blocks-bridge' } else { 'windows-blocks' }
try {
    $SimProcess = Start-Process -FilePath $SimExe -ArgumentList $SimArguments -WorkingDirectory (Split-Path $SimExe) -WindowStyle Hidden -PassThru
    $LaunchTime = Get-Date
    $RpcStarted = $false
    while (((Get-Date) - $LaunchTime).TotalSeconds -lt 75) {
        $SimProcess.Refresh()
        if ($SimProcess.HasExited) { break }
        $GpuUsed = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim()
        $GpuCounters = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object Name -Like "pid_$($SimProcess.Id)_*")
        $Dedicated = ($GpuCounters | Measure-Object DedicatedUsage -Sum).Sum
        $Shared = ($GpuCounters | Measure-Object SharedUsage -Sum).Sum
        $Samples.Add([pscustomobject]@{
            Seconds = ((Get-Date) - $LaunchTime).TotalSeconds
            WorkingSetMiB = $SimProcess.WorkingSet64 / 1MB
            PrivateBytesMiB = $SimProcess.PrivateMemorySize64 / 1MB
            DedicatedGpuMiB = $Dedicated / 1MB
            SharedGpuMiB = $Shared / 1MB
            GlobalGpuUsedMiB = [int]$GpuUsed
        })
        if ($WslBridgeProcess) {
            $WslBridgeProcess.Refresh()
            if ($WslBridgeProcess.HasExited) {
                $ProbeExit = $WslBridgeProcess.ExitCode
                break
            }
        }
        if (-not $RpcStarted -and (Get-NetTCPConnection -State Listen -LocalPort 41461 -ErrorAction SilentlyContinue)) {
            $RpcStarted = $true
            if ($UseReverseBridge) {
                $WslBridgeProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d','Ubuntu','--','/home/gyuhan/uav-vla-smoke/gate2/bin/python','/mnt/c/Users/leegy/Desktop/drone/scripts/gate2_reverse_bridge.py') -WindowStyle Hidden -PassThru -RedirectStandardOutput "$EvidenceRoot\windows-blocks-bridge-rpc.log" -RedirectStandardError "$EvidenceRoot\windows-blocks-bridge-stderr.log"
                $ReadyDeadline = (Get-Date).AddSeconds(10)
                while ((Get-Date) -lt $ReadyDeadline -and -not ((Get-Content -LiteralPath "$EvidenceRoot\windows-blocks-bridge-rpc.log" -Raw -ErrorAction SilentlyContinue) -match 'listeners ready')) { Start-Sleep -Milliseconds 200 }
                $WindowsBridgeProcess = Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList @('-NoProfile','-File',"$TaskRoot\scripts\gate2_reverse_bridge.ps1") -WindowStyle Hidden -PassThru -RedirectStandardOutput "$EvidenceRoot\windows-blocks-bridge-windows.log" -RedirectStandardError "$EvidenceRoot\windows-blocks-bridge-windows-stderr.log"
            } else {
                & wsl -d Ubuntu -- /home/gyuhan/uav-vla-smoke/gate2/bin/python /mnt/c/Users/leegy/Desktop/drone/scripts/compatibility_gate2_rpc.py --host 192.168.160.1 --label windows-blocks --output /mnt/c/Users/leegy/Desktop/drone/outputs/compatibility 2>&1 | Tee-Object -FilePath "$EvidenceRoot\windows-blocks-rpc.log"
                $ProbeExit = $LASTEXITCODE
                if ($ProbeExit -eq 0) { break }
            }
        }
        Start-Sleep -Milliseconds 500
    }
} finally {
    $SimExit = $null
    if ($SimProcess) {
        $SimProcess.Refresh()
        if ($SimProcess.HasExited) { $SimExit = $SimProcess.ExitCode }
        else { Stop-Process -Id $SimProcess.Id }
    }
    foreach ($Helper in @($WslBridgeProcess,$WindowsBridgeProcess)) {
        if ($Helper -and -not $Helper.HasExited) { Stop-Process -Id $Helper.Id }
    }
    [pscustomobject]@{
        Executable = $SimExe; Arguments = $SimArguments; Pid = $SimProcess.Id
        BaselineGlobalGpuMiB = [int]$GpuBaseline; RpcProbeExit = $ProbeExit; ExitBeforeCleanup = $SimExit
        PeakWorkingSetMiB = ($Samples | Measure-Object WorkingSetMiB -Maximum).Maximum
        PeakPrivateBytesMiB = ($Samples | Measure-Object PrivateBytesMiB -Maximum).Maximum
        PeakDedicatedGpuMiB = ($Samples | Measure-Object DedicatedGpuMiB -Maximum).Maximum
        PeakSharedGpuMiB = ($Samples | Measure-Object SharedGpuMiB -Maximum).Maximum
        PeakGlobalGpuMiB = ($Samples | Measure-Object GlobalGpuUsedMiB -Maximum).Maximum
        Samples = $Samples
        Topology = if ($UseReverseBridge) { 'Windows simulator + WSL client through temporary reverse TCP bridge' } else { 'Windows simulator + WSL client direct NAT host IP' }
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath "$EvidenceRoot\$RunLabel-launch.json" -Encoding utf8
}
