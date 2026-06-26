$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$nodeCandidates = Get-ChildItem -Path "$env:LOCALAPPDATA\OpenAI\Codex\runtimes\cua_node" -Recurse -Filter node.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
$nodeExe = $nodeCandidates | Select-Object -First 1

if (-not $nodeExe) {
  throw "Node runtime not found under $env:LOCALAPPDATA\OpenAI\Codex\runtimes\cua_node"
}

& $nodeExe "frontend-server.js"
