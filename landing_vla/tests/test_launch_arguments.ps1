$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $taskRoot 'scripts/native_process_args.ps1')
$taskPython=Join-Path $taskRoot 'assets/projectairsim-env/Scripts/python.exe'
$taskTemp=Join-Path ([IO.Path]::GetTempPath()) ('uav argv '+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskTemp | Out-Null
$taskScript=Join-Path $taskTemp 'argument probe.py'
$taskStdout=Join-Path $taskTemp 'arguments.json'
$taskStderr=Join-Path $taskTemp 'errors.txt'
try {
    [IO.File]::WriteAllText($taskScript,'import json,sys;print(json.dumps(sys.argv[1:]))')
    $taskExpected=@((Join-Path $taskTemp 'folder with spaces/worker.py'),'--host','127.0.0.1','quote " inside','trailing\')
    $taskCommand=Join-NativeArguments (@($taskScript)+$taskExpected)
    $taskProcess=Start-Process -FilePath $taskPython -ArgumentList $taskCommand -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr
    if ($taskProcess.ExitCode -ne 0) { throw 'Argument probe failed' }
    $taskActual=Get-Content $taskStdout -Raw | ConvertFrom-Json
    if ($taskActual.Count -ne $taskExpected.Count) { throw 'Argument count changed' }
    for ($taskIndex=0;$taskIndex -lt $taskExpected.Count;$taskIndex++) {
        if ($taskActual[$taskIndex] -cne $taskExpected[$taskIndex]) { throw "Argument $taskIndex changed" }
    }
    Write-Output 'Native argument boundaries PASS: spaces, quotes, trailing backslash'
    $taskWslScript=(& wsl -d Ubuntu --exec wslpath -u $taskScript.Replace('\','/')).Trim()
    $taskWslExpected=@($taskWslScript,'WSL value with spaces','--host','127.0.0.1')
    $taskWslArgs=@('-d','Ubuntu','--exec','/usr/bin/python3',$taskWslScript)+$taskWslExpected
    $taskProcess=Start-Process -FilePath 'wsl.exe' -ArgumentList (Join-NativeArguments $taskWslArgs) -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr
    if ($taskProcess.ExitCode -ne 0) { throw 'WSL argument probe failed' }
    $taskActual=Get-Content $taskStdout -Raw | ConvertFrom-Json
    if ($taskActual.Count -ne $taskWslExpected.Count) { throw 'WSL argument count changed' }
    for ($taskIndex=0;$taskIndex -lt $taskWslExpected.Count;$taskIndex++) {
        if ($taskActual[$taskIndex] -cne $taskWslExpected[$taskIndex]) { throw "WSL argument $taskIndex changed" }
    }
    Write-Output 'WSL exec argument boundaries PASS: script path and value with spaces'
} finally {
    foreach ($taskFile in @($taskScript,$taskStdout,$taskStderr)) {
        if (Test-Path -LiteralPath $taskFile) { Remove-Item -LiteralPath $taskFile }
    }
    Remove-Item -LiteralPath $taskTemp
}
