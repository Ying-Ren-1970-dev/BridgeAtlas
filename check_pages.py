import pdfplumber
import sys

pdf_path = sys.argv[1]
search_term = sys.argv[2] if len(sys.argv) > 2 else "CIDH"

with pdfplumber.open(pdf_path) as pdf:
    print(f"Total pages in PDF: {len(pdf.pages)}")
    print(f"\nPages containing '{search_term}':")
    for i in range(len(pdf.pages)):
        try:
            text = pdf.pages[i].extract_text()
            if text and search_term.upper() in text.upper():
                print(f"\nPage {i+1}:")
                # Show snippet around the search term
                lines = text.split('\n')
                for line in lines:
                    if search_term.upper() in line.upper():
                        print(f"  {line[:100]}")
        except Exception as e:
            print(f"Error on page {i+1}: {e}")
