$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$uv = Get-Command uv -ErrorAction Stop
$npm = Get-Command npm -ErrorAction Stop
$stateDir = Join-Path $projectRoot "state"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

$backendOut = Join-Path $stateDir "backend.stdout.log"
$backendErr = Join-Path $stateDir "backend.stderr.log"
$backend = Start-Process -FilePath $uv.Source -ArgumentList @(
  "run", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"
) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru   -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr

try {
  Write-Host "Backend starting at http://127.0.0.1:8000 (logs: $stateDir)"
  Write-Host "Frontend starting at http://127.0.0.1:3000"
  & $npm.Source start
}
finally {
  if ($backend -and -not $backend.HasExited) {
    Stop-Process -Id $backend.Id
  }
}
