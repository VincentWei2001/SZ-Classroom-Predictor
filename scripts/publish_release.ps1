# Push main and upload school_app_secure_portable.zip as GitHub Release v1.0.0 (or set env TAG).
# Prerequisites: Git auth to GitHub (HTTPS token or SSH), network to github.com,
# and GITHUB_TOKEN with repo scope for API upload (classic PAT or fine-grained Contents write + Metadata read).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$OwnerRepo = "VincentWei2001/SZ-Classroom-Predictor"
$Tag = if ($env:RELEASE_TAG) { $env:RELEASE_TAG } else { "v1.0.0" }
$Zip = Join-Path $Root "github_release\school_app_secure_portable.zip"
if (-not (Test-Path $Zip)) {
    Write-Error "Missing $Zip — run build_secure_package.py and copy the zip into github_release\"
}

Write-Host "Pushing main..."
git push -u origin main

if (-not $env:GITHUB_TOKEN) {
    Write-Host "GITHUB_TOKEN not set — push done (if OK). Create Release manually:" -ForegroundColor Yellow
    Write-Host "  https://github.com/$OwnerRepo/releases/new"
    Write-Host "  Attach: $Zip"
    exit 0
}

$headers = @{
    "Authorization" = "Bearer $env:GITHUB_TOKEN"
    "Accept"        = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

$releaseBody = @{
    tag_name         = $Tag
    name             = $Tag
    body             = "Windows portable build. Extract zip and run school_app.exe. See README."
    draft            = $false
    prerelease       = $false
    generate_release_notes = $true
} | ConvertTo-Json

$api = "https://api.github.com/repos/$OwnerRepo/releases"
Write-Host "Creating release $Tag via API..."
$rel = Invoke-RestMethod -Uri $api -Method Post -Headers $headers -Body $releaseBody -ContentType "application/json; charset=utf-8"

$uploadUrl = $rel.upload_url -replace '\{\?name,label\}', "?name=school_app_secure_portable.zip"
Write-Host "Uploading asset..."
Invoke-RestMethod -Uri $uploadUrl -Method Post -Headers $headers -InFile $Zip -ContentType "application/zip"

Write-Host "Done: $($rel.html_url)" -ForegroundColor Green
