$ErrorActionPreference = 'Stop'
$WslLink = [System.Net.Sockets.TcpClient]::new('127.0.0.1',41500)
$SimLink = [System.Net.Sockets.TcpClient]::new('127.0.0.1',41461)
try {
    $ForwardTask = $WslLink.GetStream().CopyToAsync($SimLink.GetStream())
    $ReverseTask = $SimLink.GetStream().CopyToAsync($WslLink.GetStream())
    [System.Threading.Tasks.Task]::WaitAny([System.Threading.Tasks.Task[]]@($ForwardTask,$ReverseTask),45000) | Out-Null
} finally {
    $WslLink.Dispose()
    $SimLink.Dispose()
}
