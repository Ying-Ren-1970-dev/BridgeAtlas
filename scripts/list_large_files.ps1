# List top 50 largest files in the repo
Get-ChildItem -Path "$PSScriptRoot\.." -File -Recurse -ErrorAction SilentlyContinue |
  Sort-Object Length -Descending |
  Select-Object -First 50 |
  ForEach-Object {
    $sizeMB = [math]::Round(($_.Length / 1MB), 2)
    "{0} MB :: {1}" -f $sizeMB, $_.FullName
  }