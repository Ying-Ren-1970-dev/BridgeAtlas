import PyPDF2
from pathlib import Path

pdf_path = Path(r"Projects\Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf")

with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    
    # Extract PDF page 9 (which should contain document page 16)
    page = pdf_reader.pages[8]  # 0-indexed, so page 9 is index 8
    text = page.extract_text()
    
    print("=" * 80)
    print("PDF Page 9 Content (Document Page 16)")
    print("=" * 80)
    print(text)
