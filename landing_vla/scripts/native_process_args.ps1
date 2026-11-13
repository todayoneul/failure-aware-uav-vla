function Enable-ProcessExitTracking {
    param([System.Diagnostics.Process]$Process)
    # Cache the native handle while the child is alive. Windows PowerShell 5.1
    # otherwise loses redirected WSL exit codes after HasExited/Refresh polling.
    $null=$Process.Handle
}

function Join-NativeArguments {
    param([string[]]$Arguments)
    # Windows CRT argv quoting, not shell interpolation or JSON escaping.
    $taskQuoted=foreach ($taskArgument in $Arguments) {
        if ($taskArgument.Length -gt 0 -and $taskArgument -notmatch '[\s"]') {
            $taskArgument
            continue
        }
        $taskBuilder=[Text.StringBuilder]::new()
        [void]$taskBuilder.Append('"')
        $taskSlashes=0
        foreach ($taskCharacter in $taskArgument.ToCharArray()) {
            if ($taskCharacter -eq '\') { $taskSlashes++; continue }
            if ($taskCharacter -eq '"') {
                [void]$taskBuilder.Append(('\' * (2*$taskSlashes+1)))
                [void]$taskBuilder.Append('"')
            } else {
                [void]$taskBuilder.Append(('\' * $taskSlashes))
                [void]$taskBuilder.Append($taskCharacter)
            }
            $taskSlashes=0
        }
        [void]$taskBuilder.Append(('\' * (2*$taskSlashes)))
        [void]$taskBuilder.Append('"')
        $taskBuilder.ToString()
    }
    $taskQuoted -join ' '
}
