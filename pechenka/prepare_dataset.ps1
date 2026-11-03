# Rebuild a deterministic 80/20 split by screenshot group, preserving unpaired files.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path $PSScriptRoot).Path
$staging = Join-Path $root 'dataset_staging'
$unmatched = Join-Path $root 'dataset_unmatched'
if (Test-Path -LiteralPath $staging) {
    throw "Staging directory already exists: $staging"
}

$images = @{}
$labels = @{}
foreach ($split in 'train', 'val') {
    foreach ($image in Get-ChildItem -LiteralPath (Join-Path $root "images/$split") -File) {
        if ($image.Extension -notin '.jpg', '.jpeg', '.png', '.bmp', '.webp') { continue }
        if ($images.ContainsKey($image.BaseName)) { throw "Duplicate image stem: $($image.BaseName)" }
        $images[$image.BaseName] = $image
    }
    foreach ($label in Get-ChildItem -LiteralPath (Join-Path $root "labels/$split") -Filter '*.txt' -File) {
        if ($labels.ContainsKey($label.BaseName)) { throw "Duplicate label stem: $($label.BaseName)" }
        $labels[$label.BaseName] = $label
    }
}

$pairs = @($images.Keys | Where-Object { $labels.ContainsKey($_) })
$groups = @($pairs | ForEach-Object { $_ -replace '_aug\d+$', '' } | Sort-Object -Unique)
if ($groups.Count -lt 5) { throw 'Too few screenshot groups for an 80/20 split' }
$valGroups = @{}
for ($i = 0; $i -lt $groups.Count; $i++) {
    if ($i % 5 -eq 0) { $valGroups[$groups[$i]] = $true }
}

New-Item -ItemType Directory -Path $staging, $unmatched -Force | Out-Null
foreach ($kind in 'images', 'labels') {
    foreach ($split in 'train', 'val') {
        New-Item -ItemType Directory -Path (Join-Path $staging "$kind/$split") -Force | Out-Null
        New-Item -ItemType Directory -Path (Join-Path $unmatched "$kind/$split") -Force | Out-Null
    }
}
foreach ($stem in $images.Keys) {
    $image = $images[$stem]
    $split = Split-Path $image.DirectoryName -Leaf
    $destination = if ($labels.ContainsKey($stem)) {
        Join-Path $staging "images/$split/$($image.Name)"
    } else {
        Join-Path $unmatched "images/$split/$($image.Name)"
    }
    Move-Item -LiteralPath $image.FullName -Destination $destination
}
foreach ($stem in $labels.Keys) {
    $label = $labels[$stem]
    $split = Split-Path $label.DirectoryName -Leaf
    $destination = if ($images.ContainsKey($stem)) {
        Join-Path $staging "labels/$split/$($label.Name)"
    } else {
        Join-Path $unmatched "labels/$split/$($label.Name)"
    }
    Move-Item -LiteralPath $label.FullName -Destination $destination
}

foreach ($stem in $pairs) {
    $group = $stem -replace '_aug\d+$', ''
    $targetSplit = if ($valGroups.ContainsKey($group)) { 'val' } else { 'train' }
    foreach ($kind in 'images', 'labels') {
        $source = Get-ChildItem -LiteralPath (Join-Path $staging $kind) -Recurse -File |
            Where-Object { $_.BaseName -eq $stem } | Select-Object -First 1
        if (-not $source) { throw "Staged file missing: $kind/$stem" }
        Move-Item -LiteralPath $source.FullName -Destination (Join-Path $root "$kind/$targetSplit/$($source.Name)")
    }
}
Get-ChildItem -LiteralPath $staging -Directory -Recurse |
    Sort-Object { $_.FullName.Length } -Descending | Remove-Item
Remove-Item -LiteralPath $staging
Write-Host "Prepared $($pairs.Count) image/label pairs in $($groups.Count) screenshot groups."
Write-Host "Unpaired files preserved in $unmatched"
