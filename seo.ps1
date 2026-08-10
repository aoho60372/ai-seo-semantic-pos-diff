[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $WorkflowArguments
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExecutable = Join-Path $projectDirectory ".venv\Scripts\python.exe"
$workflowScript = Join-Path $projectDirectory "posdiff_workflow.py"

if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Project Python environment not found: $pythonExecutable. Create it with: py -3 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}
if (-not $WorkflowArguments -or $WorkflowArguments.Count -eq 0) {
    throw "Example: .\seo.ps1 analyze"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
& $pythonExecutable $workflowScript @WorkflowArguments
exit $LASTEXITCODE
