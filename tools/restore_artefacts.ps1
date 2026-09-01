# Restore the heavy artefacts from the GitHub Release into their original paths.
# Run from a fresh clone's root in PowerShell:  .\tools\restore_artefacts.ps1
$ErrorActionPreference = "Stop"
$tag = "artefacts-v1"; $out = "_release"
New-Item -ItemType Directory -Force -Path $out | Out-Null
& gh release download $tag -D $out --clobber
# verify
$bad = 0
Get-Content (Join-Path $out "MANIFEST.sha256") | ForEach-Object {
  $h, $len, $name = $_ -split "\s+", 3
  $f = Join-Path $out $name
  if ((Get-FileHash $f -Algorithm SHA256).Hash -ne $h) { Write-Host "CHECKSUM MISMATCH: $name"; $bad++ }
}
if ($bad -gt 0) { throw "$bad archive(s) failed verification" }
# extract: every zip was created from the repo root, so it unpacks into place
Get-ChildItem $out -Filter *.zip | ForEach-Object { Write-Host "extracting $($_.Name)"; & tar.exe -xf $_.FullName }
Write-Host "restored. Recreate the environment with: python -m venv venv; venv\Scripts\pip install -r requirements.txt"
