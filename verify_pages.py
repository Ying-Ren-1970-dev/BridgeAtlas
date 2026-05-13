"""Verify what content was extracted from specific pages."""
from pathlib import Path
from pdf_processor import PDFProcessor

pdf_path = Path(r"C:\Users\alexl\Desktop\xttribute\Librarian\Projects\Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf")
pages_to_check = [8, 9, 10, 21]

print("="*80)
print("VERIFYING CIDH PILE INFORMATION EXTRACTION")
print("="*80)

# Create processor with vision enabled
processor = PDFProcessor(use_vision=True)

# Extract text from the PDF
pages_text, total_pages = processor.extract_text_from_pdf(pdf_path)

for page_num in pages_to_check:
    print(f"\n{'='*80}")
    print(f"PAGE {page_num}")
    print(f"{'='*80}")
    
    if page_num in pages_text:
        content = pages_text[page_num]
        
        # Check if it has drawing analysis
        if "[DRAWING ANALYSIS]" in content:
            print("✓ Contains VISION ANALYSIS of CAD drawing")
            print("\n--- CONTENT ---")
            print(content)
        elif "[TEXT CONTENT]" in content:
            print("✓ Contains TEXT + VISION ANALYSIS")
            print("\n--- CONTENT ---")
            print(content)
        else:
            print("✓ Contains TEXT ONLY")
            print("\n--- CONTENT ---")
            print(content[:1000] + "..." if len(content) > 1000 else content)
    else:
        print("✗ No content extracted for this page")
    
    print("\n" + "-"*80)
