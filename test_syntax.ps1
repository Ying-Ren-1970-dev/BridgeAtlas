$test = "yes"

if ($test -eq "yes") {
    Write-Host "Test 1 passed"
} else {
    Write-Host "Test 1 failed"
}

if (Test-Path "Projects") {
    Write-Host "Projects folder exists"
} else {
    Write-Host "Projects folder not found"
}

Write-Host "Done"
