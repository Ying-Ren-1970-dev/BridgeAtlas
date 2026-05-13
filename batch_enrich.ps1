# Batch Enrichment Script for All Projects
# This script enriches all PDF files in the Projects folder

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '  BATCH ENRICHMENT - All Projects' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

# Get all PDF files
$pdfs = Get-ChildItem -Path 'Projects' -Filter '*.pdf'
$total = $pdfs.Count
$current = 0

Write-Host "Found $total PDF files to process" -ForegroundColor Green
Write-Host 'Estimated cost: ~$0.35 (based on GPT-4o Vision pricing)' -ForegroundColor Yellow
Write-Host ''

# Confirm before proceeding
$confirm = Read-Host 'Proceed with enrichment? (Y/N)'
if ($confirm -ne 'Y' -and $confirm -ne 'y') {
    Write-Host 'Operation cancelled.' -ForegroundColor Red
    exit
}

Write-Host ''
Write-Host 'Starting enrichment...' -ForegroundColor Green
Write-Host ''

# Process each PDF
foreach ($pdf in $pdfs) {
    $current++
    Write-Host "[$current/$total] Processing: $($pdf.Name)" -ForegroundColor Cyan
    
    # Run enrichment
    python main.py enrich $pdf.Name
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Success: $($pdf.Name)" -ForegroundColor Green
    } else {
        Write-Host "  Error: $($pdf.Name)" -ForegroundColor Red
    }
    Write-Host ''
}

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '  Enrichment Complete!' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host 'Enriched metadata saved to: data' -ForegroundColor Green
Write-Host 'View JSON files: Get-ChildItem data' -ForegroundColor Yellow
