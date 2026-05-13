import PyPDF2
from pathlib import Path

pdf_path = Path(r"Projects\Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf")

with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    total_pages = len(pdf_reader.pages)
    
    print(f"Checking pages with extractable text for CIDH content...\n")
    
    # Only check first 100 pages and skip problematic ones
    cidh_pages = []
    pages_checked = 0
    
    for page_num in range(min(total_pages, 100)):
        try:
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            # Only count pages with substantial text
            if text and text.strip() and len(text.strip()) > 50:
                pages_checked += 1
                if 'CIDH' in text.upper():
                    count = text.upper().count('CIDH')
                    cidh_pages.append(page_num + 1)
                    print(f"✓ PDF Page {page_num + 1}: {count} mentions of 'CIDH'")
        except Exception as e:
            pass
    
    print(f"\n{'='*80}")
    print(f"Checked {pages_checked} pages with extractable text")
    print(f"Found CIDH content on {len(cidh_pages)} page(s): {cidh_pages}")
    print(f"{'='*80}")
