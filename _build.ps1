# _build.ps1 - local rebuild of the Reasonix CLI into dist/.
#
# Why not `make build`: the repo root carries BOTH .git and fossil sidecar
# metadata, so any main-package build fails with "multiple VCS detected"
# unless VCS stamping is disabled (-buildvcs=false). This script pins that
# flag so local rebuilds never trip over it.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File _build.ps1            # CLI
#   powershell -NoProfile -ExecutionPolicy Bypass -File _build.ps1 -Desktop   # CLI + Wails GUI
# Output:
#   dist\reasonix.exe
#   dist\reasonix-desktop.exe   (-Desktop)
[CmdletBinding()]
param(
    # Additionally build the Wails desktop shell into dist\reasonix-desktop.exe.
    [switch]$Desktop
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Get-GitOutput([string]$Argsline, [string]$Fallback) {
    $out = & git $Argsline.Split(' ') 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($out)) { return $Fallback }
    return ([string]$out).Trim()
}

$version   = Get-GitOutput 'describe --tags --always' 'dev'
$commit    = Get-GitOutput 'rev-parse --short=12 HEAD' 'unknown'
$buildTime = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

# Mirrors the Makefile LDFLAGS (main.version / main.gitCommit / main.buildTimeUTC).
$ldflags = "-s -w -X main.version=$version -X main.gitCommit=$commit -X main.buildTimeUTC=$buildTime"

New-Item -ItemType Directory -Force -Path dist | Out-Null

Write-Host "== build cli -> dist\reasonix.exe (version=$version commit=$commit)"
& go build -buildvcs=false -trimpath -ldflags $ldflags -o dist/reasonix.exe ./cmd/reasonix
if ($LASTEXITCODE -ne 0) {
    Write-Error "go build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

$exe = Resolve-Path dist/reasonix.exe
Write-Host "== ok: $exe ($([math]::Round((Get-Item $exe).Length / 1MB, 2)) MB)"

if ($Desktop) {
    # Prefer PATH; fall back to the default go install location so a
    # GOPATH\bin-less PATH never blocks local rebuilds.
    $wails = (Get-Command wails -ErrorAction SilentlyContinue).Source
    if (-not $wails) {
        $wails = Join-Path $env:USERPROFILE 'go\bin\wails.exe'
    }
    if (-not (Test-Path $wails)) {
        Write-Error "wails CLI not found; install with: go install github.com/wailsapp/wails/v2/cmd/wails@v2.13.0"
        exit 1
    }

    # Same flags as CI (ci.yml desktop job): fast, plain exe, no NSIS packaging.
    # GOFLAGS carries -buildvcs=false into the go build wails spawns internally,
    # which would otherwise trip over the fossil sidecar next to .git.
    $env:GOFLAGS = '-buildvcs=false'
    $desktopLdflags = "-X main.version=$version -X main.channel=stable"

    # Build the React frontend first (mirrors ci.yml:385): wails runs with -s
    # (-skipfrontend), so without this the //go:embed of frontend/dist bakes an
    # empty FS and the app dies at startup with "no index.html could be found".
    Write-Host "== build desktop frontend (pnpm)"
    Push-Location desktop/frontend
    try {
        & pnpm install --config.confirmModulesPurge=false
        if ($LASTEXITCODE -ne 0) {
            Write-Error "pnpm install failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        & pnpm build
        if ($LASTEXITCODE -ne 0) {
            Write-Error "pnpm build failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
    finally {
        Pop-Location
    }

    Push-Location desktop
    try {
        Write-Host "== build desktop -> dist\reasonix-desktop.exe (via $wails)"
        & $wails build -clean -s -skipbindings -nopackage -platform windows/amd64 -webview2 embed -ldflags $desktopLdflags
        if ($LASTEXITCODE -ne 0) {
            Write-Error "wails build failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
    finally {
        Pop-Location
    }

    Copy-Item desktop/build/bin/reasonix-desktop.exe dist/ -Force
    $gui = Resolve-Path dist/reasonix-desktop.exe
    Write-Host "== ok: $gui ($([math]::Round((Get-Item $gui).Length / 1MB, 2)) MB)"
}
