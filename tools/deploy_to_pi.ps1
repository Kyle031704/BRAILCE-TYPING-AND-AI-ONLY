<#
.SYNOPSIS
    Brailce Deployment Helper Script (Windows PowerShell)
.DESCRIPTION
    Pushes local code changes to GitHub repository so the Raspberry Pi can pull them,
    or directly transfers files via SCP if configured.
#>

param(
    [switch]$DirectScp,
    [string]$PiHost = $env:PI_IP,
    [string]$PiUser = $(if ($env:PI_USER) { $env:PI_USER } else { "pi" })
)

$ErrorActionPreference = "Stop"
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Brailce Windows -> Raspberry Pi Deploy" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Ensure local git repository is committed
$status = git status --porcelain
if ($status) {
    Write-Host "[!] You have uncommitted changes. Staging and committing..." -ForegroundColor Yellow
    git add -A
    git commit -m "feat: deploy latest Brailce updates"
} else {
    Write-Host "[OK] Working tree clean." -ForegroundColor Green
}

# Step 2: Push to GitHub
Write-Host "`n[*] Pushing latest commits to GitHub (origin master)..." -ForegroundColor Cyan
git push origin master
Write-Host "[OK] GitHub repository is now completely up to date!" -ForegroundColor Green

if ($DirectScp -and $PiHost) {
    Write-Host "`n[*] Direct SCP mode requested to $PiUser@$PiHost..." -ForegroundColor Cyan
    scp -r brailce tools requirements.txt requirements-pi.txt main.py "${PiUser}@${PiHost}:~/brailce/"
    Write-Host "[OK] Files copied directly via SCP!" -ForegroundColor Green
} else {
    Write-Host "`nNext Step: Run these commands in PuTTY on your Raspberry Pi:" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------------" -ForegroundColor Gray
    Write-Host "cd ~/brailce" -ForegroundColor White
    Write-Host "git pull origin master" -ForegroundColor White
    Write-Host "python3 tools/audit_audio.py" -ForegroundColor White
    Write-Host "python3 main.py" -ForegroundColor White
    Write-Host "--------------------------------------------------------" -ForegroundColor Gray
}
