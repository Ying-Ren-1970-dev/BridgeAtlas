"""Test vision analysis on a single page."""
from pathlib import Path
from vision_analyzer import VisionAnalyzer

# Test on page 9 of Mar Vista (the CIDH table page)
pdf_path = Path(r"C:\Users\alexl\Desktop\xttribute\Librarian\Projects\Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf")
page_num = 9

print(f"Testing vision analysis on {pdf_path.name}, page {page_num}...")

analyzer = VisionAnalyzer()
description = analyzer.analyze_pdf_page(pdf_path, page_num)

if description:
    print("\n" + "="*80)
    print("VISION ANALYSIS RESULT:")
    print("="*80)
    print(description)
    print("="*80)
else:
    print("Vision analysis failed!")
