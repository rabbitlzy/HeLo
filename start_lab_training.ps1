param(
    [string]$PythonExe = 'C:\conda_envs\helo4090\python.exe',
    [string]$RunName = 'dmer_code_73_4090'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Python not found: $PythonExe" }
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must contain only letters, numbers, underscore or hyphen.' }
$taskOutput = Join-Path $PSScriptRoot "runs\$RunName"
$taskLogs = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$taskArguments = @('-u', (Join-Path $PSScriptRoot 'run_reproduction.py'), '--implementation', 'code-repaired', '--output', $taskOutput)
if (Test-Path -LiteralPath (Join-Path $taskOutput 'manifest.json')) { $taskArguments += '--resume' }
# The scheduler must be set to disallow concurrent instances of this task.
$taskQuotedArguments = $taskArguments | ForEach-Object { '"' + $_ + '"' }
$taskProcess = Start-Process -FilePath $PythonExe -ArgumentList $taskQuotedArguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput (Join-Path $taskLogs "$RunName`_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskLogs "$RunName`_$taskStamp.stderr.log")
exit $taskProcess.ExitCode
