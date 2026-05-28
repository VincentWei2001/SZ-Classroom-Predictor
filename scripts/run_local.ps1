# 本地启动（需 conda 环境 school_app）
# 打包发布: python scripts\build_school_app.py 后运行 scripts\build_portable_package.py
$Root = Split-Path -Parent $PSScriptRoot
$Python = "D:\Anaconda\envs\school_app\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "未找到 $Python，请安装或修改本脚本中的 Python 路径。"
}
Set-Location $Root
& $Python "src\school_app_portable.py"
