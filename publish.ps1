# Publishes this project to a PUBLIC GitHub repo named smokies-forecast.
# Run from a normal (non-admin) PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Documents\portfolio\01-smokies-forecast\publish.ps1"
# Safe to re-run: it reuses the repo and pushes new commits.

Set-Location $PSScriptRoot
$RepoName = 'smokies-forecast'
$Desc = 'Forecasting monthly visits to Great Smoky Mountains National Park with NPS, weather and event data'

function Need($cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "$cmd isn't installed or isn't on PATH. Close and reopen PowerShell, then try again." -ForegroundColor Red
        exit 1
    }
}
Need git; Need gh

# 1. Git identity (used as the commit author)
if (-not (git config --global user.name))  { git config --global user.name  (Read-Host 'Name to show on your commits (e.g. Jim Smith)') }
if (-not (git config --global user.email)) { git config --global user.email (Read-Host 'Email for commits (use your GitHub email or its noreply address)') }

# 2. Sign in to GitHub (opens your browser; you approve there)
gh auth status 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nSigning in to GitHub. Pick: GitHub.com > HTTPS > Login with a web browser." -ForegroundColor Cyan
    gh auth login --hostname github.com --git-protocol https --web
    if ($LASTEXITCODE -ne 0) { Write-Host 'GitHub sign-in did not finish.' -ForegroundColor Red; exit 1 }
}
gh auth setup-git | Out-Null

# 3. Commit
if (-not (Test-Path .git)) { git init -b main | Out-Null }
git add -A
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -q -m 'Smokies visitation forecast: data, models, weather and event analysis, Streamlit app' `
        -m 'NPS monthly visits 1979-2026, Open-Meteo weather, dated park disruptions, walk-forward backtests, and a Streamlit app.' `
        -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`nClaude-Session: https://claude.ai/code/session_01PfHFoCkjNUvnoyJnkeFQXP"
}

# 4. Create the public repo (first run) or push (later runs)
$user = gh api user --jq .login
gh repo view "$user/$RepoName" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    $ok = Read-Host "`nThis creates a PUBLIC repo github.com/$user/$RepoName. Type yes to continue"
    if ($ok -ne 'yes') { Write-Host 'Stopped. Nothing was published.'; exit 0 }
    gh repo create $RepoName --public --description $Desc --source . --remote origin --push
} else {
    if (-not (git remote)) { git remote add origin "https://github.com/$user/$RepoName.git" }
    git push -u origin main
}
if ($LASTEXITCODE -ne 0) { Write-Host 'Push failed. Copy the error above to Claude.' -ForegroundColor Red; exit 1 }

Write-Host "`nPublished: https://github.com/$user/$RepoName" -ForegroundColor Green
gh repo view "$user/$RepoName" --web
