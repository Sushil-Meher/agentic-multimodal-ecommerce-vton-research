# Local database helper (Windows PowerShell).
#   .\scripts\db.ps1 up       start Postgres; restore latest backup, else rebuild from the SQL files
#   .\scripts\db.ps1 backup   dump the database to the backup folder
#   .\scripts\db.ps1 restore  restore the latest backup (or -File <path>)
# Backups live outside the repo and outside Docker, so a Docker reinstall does not remove them.
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet("up", "backup", "restore")][string]$Command,
    [string]$File = ""
)

# docker prints progress to stderr, so rely on exit codes rather than stopping on stderr
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$container = "sambalpuri-ai-postgres"
$dbName = if ($env:POSTGRES_DB) { $env:POSTGRES_DB } else { "sambalpuri_ai" }
$dbUser = if ($env:POSTGRES_USER) { $env:POSTGRES_USER } else { "app_user" }
$backupDir = if ($env:DB_BACKUP_DIR) { $env:DB_BACKUP_DIR } else { "D:\backups\sambalpuri_ai" }
$sqlFiles = @(
    "docs\database\schema.sql",
    "docs\database\seed.sql",
    "experiments\benchmarks\seed_eval_catalog.sql"
)

function Psql-File($hostPath) {
    docker cp $hostPath "${container}:/tmp/job.sql" | Out-Null
    docker exec $container psql -v ON_ERROR_STOP=1 -q -U $dbUser -d $dbName -f /tmp/job.sql
    if ($LASTEXITCODE -ne 0) { throw "psql failed for $hostPath" }
}

function Wait-Ready {
    for ($i = 0; $i -lt 60; $i++) {
        docker exec $container pg_isready -U $dbUser -d $dbName 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return }
        Start-Sleep -Seconds 2
    }
    throw "Postgres did not become ready"
}

function Latest-Backup {
    if (-not (Test-Path $backupDir)) { return $null }
    Get-ChildItem $backupDir -Filter "*.sql" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

function Do-Restore($path) {
    Write-Host "restoring $path"
    Psql-File $path
}

function Do-Backup {
    New-Item -ItemType Directory -Force $backupDir | Out-Null
    $out = Join-Path $backupDir ("{0}_{1}.sql" -f $dbName, (Get-Date -Format "yyyyMMdd_HHmmss"))
    docker exec $container pg_dump -U $dbUser -d $dbName --clean --if-exists --no-owner -f /tmp/backup.sql
    if ($LASTEXITCODE -ne 0) { throw "pg_dump failed" }
    docker cp "${container}:/tmp/backup.sql" $out | Out-Null
    Write-Host "backup written: $out"
}

switch ($Command) {
    "up" {
        docker compose -f (Join-Path $root "deployment\docker-compose.yml") up -d
        if ($LASTEXITCODE -ne 0) { throw "docker compose failed" }
        Wait-Ready
        $has = docker exec $container psql -U $dbUser -d $dbName -tAc "SELECT to_regclass('catalog.products') IS NOT NULL"
        if ($has.Trim() -eq "t") { Write-Host "database already set up"; break }
        $latest = Latest-Backup
        if ($latest) { Do-Restore $latest.FullName }
        else {
            foreach ($f in $sqlFiles) { Write-Host "applying $f"; Psql-File (Join-Path $root $f) }
        }
        Write-Host "database ready"
    }
    "backup" { Wait-Ready; Do-Backup }
    "restore" {
        Wait-Ready
        $path = $File
        if (-not $path) {
            $latest = Latest-Backup
            if (-not $latest) { throw "no backup found in $backupDir" }
            $path = $latest.FullName
        }
        Do-Restore $path
    }
}
