param(
    [Parameter(Mandatory = $true)]
    [int]$StartYear,
    [Parameter(Mandatory = $true)]
    [int]$EndYear,
    [Parameter(Mandatory = $true)]
    [string]$ArchiveDir,
    [string]$RawDir = ".\data\global_surface\raw",
    [string]$ProcessedDir = ".\data\global_surface\processed",
    [string[]]$Stations
)

$ErrorActionPreference = "Stop"
$pipeline = Join-Path $PSScriptRoot "..\data_fcl_pipeline"

if ($EndYear -lt $StartYear) {
    throw "EndYear must be greater than or equal to StartYear."
}

python -m pipeline.global_surface.download_archives `
    --start-year $StartYear --end-year $EndYear --archive-dir $ArchiveDir `
    --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$extractArgs = @(
    "-m", "pipeline.global_surface.archive", "extract",
    "--archive-dir", $ArchiveDir,
    "--start-year", $StartYear,
    "--end-year", $EndYear,
    "--raw-dir", $RawDir
)
if ($Stations -and $Stations.Count -gt 0) {
    $extractArgs += "--stations"
    $extractArgs += $Stations
}
python @extractArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m pipeline.global_surface.season_tasks `
    --raw-dir $RawDir --output-dir $ProcessedDir `
    --features TEMP --input-days 31 --forecast-days 7
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "Prepared GSOD seasonal tasks in $ProcessedDir"
