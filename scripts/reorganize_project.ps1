# One-time / idempotent project layout cleanup. Run from repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\reorganize_project.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Ensure-Dir($rel) {
    $p = Join-Path $Root $rel
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
}

function Move-IfExists($from, $toDir) {
    $src = Join-Path $Root $from
    if (-not (Test-Path $src)) { return }
    Ensure-Dir $toDir
    $dest = Join-Path (Join-Path $Root $toDir) (Split-Path $src -Leaf)
    if ($src -eq $dest) { return }
    if (Test-Path $dest) { return }
    Move-Item -LiteralPath $src -Destination $dest
    Write-Host "  $from -> $toDir\"
}

Write-Host "Creating directories..."
@(
    "app", "app\notebooks", "src", "models", "data\csv", "archive\notebooks",
    "archive\exports", "archive\logs", "archive\specs", "release", "docs"
) | ForEach-Object { Ensure-Dir $_ }

Write-Host "Moving model folders..."
Get-ChildItem -LiteralPath $Root -Directory | Where-Object {
    $_.Name -match '^\d{4}_' -or $_.Name -match '^0312_.*_replaced$'
} | ForEach-Object {
    $dest = Join-Path (Join-Path $Root "models") $_.Name
    if (-not (Test-Path $dest)) {
        Move-Item -LiteralPath $_.FullName -Destination $dest
        Write-Host "  $($_.Name) -> models\"
    }
}

Write-Host "Moving CSV data..."
Get-ChildItem -LiteralPath $Root -File -Filter "*.csv" | ForEach-Object {
    $dest = Join-Path (Join-Path $Root "data\csv") $_.Name
    if (-not (Test-Path $dest)) {
        Move-Item -LiteralPath $_.FullName -Destination $dest
        Write-Host "  $($_.Name) -> data\csv\"
    }
}

Write-Host "Moving source files..."
@(
    "school_app_portable.py", "secure_model_bundle.py", "sklearn_joblib_compat.py"
) | ForEach-Object { Move-IfExists $_ "src" }

Write-Host "Moving build scripts..."
@(
    "build_school_app.py", "build_secure_package.py", "build_light_package.py", "prepare_min_release.py"
) | ForEach-Object {
    $srcPath = Join-Path $Root $_
    $destPath = Join-Path (Join-Path $Root "scripts") $_
    if ((Test-Path $srcPath) -and -not (Test-Path $destPath)) {
        Move-Item -LiteralPath $srcPath -Destination $destPath
        Write-Host "  $_ -> scripts\"
    }
}

Write-Host "Moving notebooks..."
@("classroom_predictor_app.ipynb") | ForEach-Object { Move-IfExists $_ "app" }
@(
    "预测应用完整版.ipynb", "预测应用完整版 copy.ipynb", "预测应用.ipynb",
    "预测应用_Slider.ipynb", "对比分析.ipynb", "堆叠式集成（stacking）XGboost 2026.ipynb"
) | ForEach-Object {
    $src = Join-Path $Root $_
    if (Test-Path $src) {
        $dest = Join-Path (Join-Path $Root "app\notebooks") $_
        if (-not (Test-Path $dest)) {
            Ensure-Dir "app\notebooks"
            Move-Item -LiteralPath $src -Destination $dest
            Write-Host "  $_ -> app\notebooks\"
        }
    }
}

Write-Host "Moving release artifacts..."
if (Test-Path (Join-Path $Root "github_release")) {
    Get-ChildItem (Join-Path $Root "github_release") | ForEach-Object {
        $dest = Join-Path (Join-Path $Root "release") $_.Name
        if (-not (Test-Path $dest)) {
            Move-Item -LiteralPath $_.FullName -Destination $dest
        }
    }
    Remove-Item (Join-Path $Root "github_release") -Recurse -Force -ErrorAction SilentlyContinue
}

Move-IfExists "school_app_models.bin" "models"
Move-IfExists "school_app_secure_portable.zip" "release"

Write-Host "Moving archive clutter..."
@("classroom_model.obj", "classroom_model.mtl", "classroom_model.3ds",
  "classroom_model2.obj", "classroom_model2.mtl") | ForEach-Object { Move-IfExists $_ "archive\exports" }
@("build_secure_package.log") | ForEach-Object { Move-IfExists $_ "archive\logs" }
@("school_app_debug.spec", "school_app_light.spec") | ForEach-Object { Move-IfExists $_ "archive\specs" }
# school_app.spec stays at root (PyInstaller regenerates here)

Write-Host "Done. See docs\PROJECT_LAYOUT.md"
