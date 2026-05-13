import PyPDF2
from pathlib import Path

pdf_path = Path(r"Projects\Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf")

with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    total_pages = len(pdf_reader.pages)
    print(f"Total pages in PDF: {total_pages}")
    print(f"\nPages with extractable text:")
    
    pages_with_text = []
    for page_num in range(min(total_pages, 100)):  # Check first 100 pages
        try:
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text and text.strip() and len(text.strip()) > 50:
                pages_with_text.append(page_num + 1)  # 1-indexed
                # Check if this page mentions CIDH
                if 'CIDH' in text.upper():
                    print(f"  Page {page_num + 1}: HAS TEXT (includes 'CIDH')")
                elif page_num < 10 or (page_num + 1) in [15, 16, 17, 18, 19, 20]:
                    print(f"  Page {page_num + 1}: has text")
        except Exception as e:
            pass
    
    print(f"\nTotal pages with extractable text: {len(pages_with_text)}")
    print(f"Page numbers: {pages_with_text}")
