import PyPDF2
from pathlib import Path

pdf_path = Path(r"Projects\Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf")

with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    total_pages = len(pdf_reader.pages)
    
    print(f"Scanning all {total_pages} pages for CIDH content...\n")
    
    cidh_pages = []
    for page_num in range(total_pages):
        try:
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text and 'CIDH' in text.upper():
                cidh_pages.append(page_num + 1)  # 1-indexed
                # Count occurrences
                count = text.upper().count('CIDH')
                print(f"PDF Page {page_num + 1}: Found {count} mentions of 'CIDH'")
        except Exception as e:
            pass
    
    print(f"\n{'='*80}")
    print(f"Summary: Found CIDH content on {len(cidh_pages)} pages")
    print(f"PDF Page numbers: {cidh_pages}")
    print(f"{'='*80}")
