# Archive the git-ignored heavy artefacts of msc-thesis as GitHub Release assets.
# Run from the repo root in PowerShell:  .\tools\archive_artefacts.ps1
# Requires: GitHub CLI (winget install GitHub.cli) and a one-time `gh auth login`.
# Each archive stays under GitHub's 2 GB per-asset limit; the SHA-256 manifest is uploaded with them.
$ErrorActionPreference = "Stop"
$tag = "artefacts-v1"
$out = "_release"
New-Item -ItemType Directory -Force -Path $out | Out-Null

# (archive name, source path) pairs. tar.exe (bsdtar, built into Windows 10+) writes zip via -a.
$items = @(
  @("data_raw.zip",              "data/raw"),
  @("data_cache.zip",            "data/cache"),
  @("data_processed.zip",        "data/processed"),
  @("data_embeddings_glove.zip", "data/embeddings"),
  @("preproc_ablation_cache.zip","results/preproc_ablation/cache")
)
Get-ChildItem "results/transformers/checkpoints" -Directory | ForEach-Object {
  $items += ,@("ckpt256__$($_.Name).zip", "results/transformers/checkpoints/$($_.Name)")
}
Get-ChildItem "results/transformers_512/checkpoints" -Directory | ForEach-Object {
  $items += ,@("ckpt512__$($_.Name).zip", "results/transformers_512/checkpoints/$($_.Name)")
}
if (Test-Path "results/transformers/_partial")     { $items += ,@("transformers256_partial.zip", "results/transformers/_partial") }
if (Test-Path "results/transformers_512/_partial") { $items += ,@("transformers512_partial.zip", "results/transformers_512/_partial") }

foreach ($it in $items) {
  $zip = Join-Path $out $it[0]; $src = $it[1]
  if (-not (Test-Path $src)) { Write-Host "skip (missing): $src"; continue }
  if (Test-Path $zip) { Write-Host "exists: $zip"; continue }
  Write-Host "zipping $src -> $zip"
  & tar.exe -a -cf $zip $src
}

# SHA-256 manifest (restore script verifies against it)
$manifest = Join-Path $out "MANIFEST.sha256"
Get-ChildItem $out -Filter *.zip | ForEach-Object {
  "{0}  {1}  {2}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.Length, $_.Name
} | Set-Content $manifest
Write-Host "manifest written: $manifest"

# Create the release (idempotent: reuse if it exists) and upload everything.
# NOTE: `gh release view` writes to stderr when the tag is absent; under PowerShell 5.1 with
# $ErrorActionPreference='Stop' that becomes a terminating NativeCommandError and kills the script
# right after zipping (observed 2026-09-01). `gh release list` exits 0 and prints nothing instead.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$tags = @(& gh release list --json tagName --jq ".[].tagName")
if ($tags -notcontains $tag) {
  Write-Host "creating release $tag"
  & gh release create $tag --title "Heavy artefacts (checkpoints, caches, embeddings, raw data)" --notes-file ARTEFACTS.md
} else {
  Write-Host "release $tag already exists; reusing"
}

$failed = @()
Get-ChildItem $out -File | ForEach-Object {
  Write-Host ("uploading {0} ({1} MB)" -f $_.Name, [math]::Round($_.Length/1MB,0))
  & gh release upload $tag $_.FullName --clobber
  if ($LASTEXITCODE -ne 0) { $failed += $_.Name }
}
$ErrorActionPreference = $prevEAP

if ($failed.Count -gt 0) {
  Write-Host ("FAILED uploads ({0}): {1}" -f $failed.Count, ($failed -join ", "))
  Write-Host "Re-run this script: existing zips are skipped and uploads use --clobber."
  exit 1
}
Write-Host "done: https://github.com/akinaydin0/msc-thesis/releases/tag/$tag"
