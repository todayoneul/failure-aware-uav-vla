param([int]$ReversePort,[int]$SimulatorPort,[int]$DurationSeconds=7200)
$ErrorActionPreference = 'Stop'
$taskDeadline = (Get-Date).AddSeconds($DurationSeconds)
while((Get-Date) -lt $taskDeadline) {
    $taskWslLink = $null; $taskSimLink = $null
    try {
        $taskWslLink = [System.Net.Sockets.TcpClient]::new('127.0.0.1',$ReversePort)
        $taskSimLink = [System.Net.Sockets.TcpClient]::new('127.0.0.1',$SimulatorPort)
        $taskWslLink.NoDelay = $true; $taskSimLink.NoDelay = $true
        $taskForward = $taskWslLink.GetStream().CopyToAsync($taskSimLink.GetStream())
        $taskBackward = $taskSimLink.GetStream().CopyToAsync($taskWslLink.GetStream())
        $null = [System.Threading.Tasks.Task]::WaitAny([System.Threading.Tasks.Task[]]@($taskForward,$taskBackward),$DurationSeconds*1000)
    } catch {
        Write-Output $_.Exception.Message
    } finally {
        if($taskWslLink) { $taskWslLink.Dispose() }
        if($taskSimLink) { $taskSimLink.Dispose() }
    }
    Start-Sleep -Milliseconds 200
}
