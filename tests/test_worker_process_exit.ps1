$ErrorActionPreference='Stop'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts/native_process_args.ps1')
$taskTemp=Join-Path ([IO.Path]::GetTempPath()) ('uav worker exit '+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskTemp | Out-Null
$taskOut=Join-Path $taskTemp 'stdout.txt'
$taskErr=Join-Path $taskTemp 'stderr.txt'
$taskProcess=$null
try {
    $taskProcess=Start-Process -FilePath 'wsl.exe' -ArgumentList (Join-NativeArguments @('-d','Ubuntu','--exec','/usr/bin/sleep','0.2')) -WindowStyle Hidden -PassThru -RedirectStandardOutput $taskOut -RedirectStandardError $taskErr
    Enable-ProcessExitTracking $taskProcess
    while (-not $taskProcess.HasExited) { Start-Sleep -Milliseconds 50; $taskProcess.Refresh() }
    if ($null -eq $taskProcess.ExitCode -or $taskProcess.ExitCode -ne 0) { throw 'Completed WSL process lost its successful exit code' }
    Write-Output "WSL exit-code tracking PASS on PowerShell $($PSVersionTable.PSVersion)"
} finally {
    if ($taskProcess -and -not $taskProcess.HasExited) { $taskProcess.WaitForExit() }
    foreach ($taskFile in @($taskOut,$taskErr)) {
        if (Test-Path -LiteralPath $taskFile) { Remove-Item -LiteralPath $taskFile }
    }
    Remove-Item -LiteralPath $taskTemp
}
