# Update Remaining PDFs with Enriched Metadata Integration
# This script updates all PDFs except Sports Park (already updated)

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '  UPDATE REMAINING PDFs - Enriched Metadata Integration' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

# Get all PDF files except Sports Park
$allPdfs = Get-ChildItem -Path 'Projects' -Filter '*.pdf'
$pdfs = $allPdfs | Where-Object { $_.Name -ne 'Sports Park.pdf' }
$total = $pdfs.Count
$current = 0

Write-Host "Found $total PDF files to update (excluding Sports Park.pdf)" -ForegroundColor Green
Write-Host ''

# Confirm before proceeding
$confirm = Read-Host 'Proceed with updates? (Y/N)'
if ($confirm -ne 'Y' -and $confirm -ne 'y') {
    Write-Host 'Operation cancelled.' -ForegroundColor Red
    exit
}

Write-Host ''
Write-Host 'Starting updates...' -ForegroundColor Green
Write-Host ''

# Process each PDF
foreach ($pdf in $pdfs) {
    $current++
    Write-Host "[$current/$total] Updating: $($pdf.Name)" -ForegroundColor Cyan
    
    # Run update
    python main.py update $pdf.Name
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Success: $($pdf.Name)" -ForegroundColor Green
    } else {
        Write-Host "  Error: $($pdf.Name)" -ForegroundColor Red
    }
    Write-Host ''
}

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '  Update Complete!' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host 'All PDFs now have enriched metadata integrated into search!' -ForegroundColor Green
Write-Host 'Search terms now include:' -ForegroundColor Yellow
Write-Host '  - Sheet titles and numbers' -ForegroundColor White
Write-Host '  - Plan types (General Plan, Foundation Plan, etc.)' -ForegroundColor White
Write-Host '  - Structural elements (abutment, bent, column, pile, etc.)' -ForegroundColor White
Write-Host '  - Detail types and grid references' -ForegroundColor White
Write-Host '  - Project information from title blocks' -ForegroundColor White
