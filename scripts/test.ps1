$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$codeDir = Join-Path $repoRoot "code"

Push-Location $codeDir
try {
    python -m py_compile `
        groundlm_serialization.py `
        run_api_models.py `
        run_qwen_local.py `
        audit_results.py `
        tests/test_protocol.py
    python -m unittest discover -s tests -v
}
finally {
    Pop-Location
}
