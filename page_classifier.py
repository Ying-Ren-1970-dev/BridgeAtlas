"""Page classification and title block extraction module."""
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List
import fitz  # PyMuPDF
from openai import OpenAI
from PIL import Image
import json

import config


class PageClassifier:
    """Classifies pages and extracts title block information."""
    
    def __init__(self):
        """Initialize the page classifier."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = "gpt-4o"
    
    def pdf_page_to_image(self, pdf_path: Path, page_num: int, max_size: int = 2048) -> Optional[bytes]:
        """
        Convert a PDF page to an image.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            max_size: Maximum dimension size
            
        Returns:
            Image bytes in PNG format or None if conversion fails
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]  # 0-indexed in fitz
            
            # Render page to pixmap
            zoom = 2  # 144 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            
            # Resize if needed
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert back to bytes
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            doc.close()
            
            return buffer.getvalue()
        
        except Exception as e:
            print(f"Error converting page {page_num} to image: {str(e)}")
            return None
    
    def classify_and_extract(
        self, 
        image_bytes: bytes, 
        page_num: int, 
        filename: str
    ) -> Dict:
        """
        Classify page type and extract title block information.
        
        Args:
            image_bytes: Image bytes in PNG format
            page_num: Page number being analyzed
            filename: Name of the PDF file
            
        Returns:
            Dictionary with classification and extracted data
        """
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create analysis prompt
            prompt = """Analyze this page from a structural engineering document.

TASK 1: Classify the page type
- **report**: Text-heavy reports, calculations, narratives, specifications, general notes
- **plan**: CAD drawings, structural plans, foundation plans, details, sections, elevations, schedules

TASK 2: If this is a PLAN (CAD drawing), extract title block information

Extract the following and return as JSON:

{
  "page_type": "report" or "plan",
  "confidence": 0.0-1.0,
  "title_block": {
    "project_info": {
      "project_name": "exact project name from title block",
      "project_number": "project/job number if shown",
      "client": "client name if shown",
      "location": "project location/address if shown",
      "engineer_on_record": "name of engineer who sealed/signed",
      "engineering_firm": "firm name",
      "project_year": "year from date or title block",
      "other_info": "any other relevant project details"
    },
    "plan_type": {
      "primary_type": "e.g., Foundation Plan, Structural Plan, General Plan, Detail Sheet, etc.",
      "sheet_title": "exact sheet title from drawing",
      "sheet_number": "sheet number (e.g., S-1, S-2, etc.)",
      "scale": "drawing scale if shown",
      "revision": "revision number/letter if shown",
      "date": "drawing date"
    },
    "plan_contents": {
      "has_tables": true/false,
      "has_diagrams": true/false,
      "structural_elements": ["list of elements: abutment, bent, column, footing, CIDH pile, etc."],
      "detail_types": ["list of detail types shown"],
      "grid_references": ["bent numbers, abutment numbers, station numbers if shown"]
    }
  }
}

IMPORTANT:
- If page_type is "report", set title_block to null
- Extract exact text from title block - don't infer or guess
- If information is not visible, use null for that field
- For structural_elements, list all visible elements (piles, footings, columns, beams, walls, caps, shear keys, etc.)
- Be precise with sheet numbers and titles

Return ONLY the JSON, no other text."""

            # Call GPT-4 Vision API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Add page number and filename
            result['page_number'] = page_num
            result['filename'] = filename
            
            return result
        
        except Exception as e:
            print(f"Error classifying page {page_num}: {str(e)}")
            return {
                'page_type': 'unknown',
                'confidence': 0.0,
                'page_number': page_num,
                'filename': filename,
                'title_block': None,
                'error': str(e)
            }
    
    def batch_classify_pdf(
        self, 
        pdf_path: Path, 
        sample_pages: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Classify multiple pages from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            sample_pages: List of page numbers to analyze (1-indexed), or None for all
            
        Returns:
            List of classification results
        """
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            # Determine which pages to analyze
            if sample_pages is None:
                # Analyze first 5, middle 5, and last 5 pages for large docs
                if total_pages <= 15:
                    pages_to_analyze = list(range(1, total_pages + 1))
                else:
                    pages_to_analyze = (
                        list(range(1, 6)) +  # First 5
                        list(range(total_pages // 2 - 2, total_pages // 2 + 3)) +  # Middle 5
                        list(range(total_pages - 4, total_pages + 1))  # Last 5
                    )
            else:
                pages_to_analyze = sample_pages
            
            results = []
            for page_num in pages_to_analyze:
                print(f"Analyzing page {page_num}/{total_pages}...")
                image_bytes = self.pdf_page_to_image(pdf_path, page_num)
                if image_bytes:
                    result = self.classify_and_extract(
                        image_bytes, 
                        page_num, 
                        pdf_path.name
                    )
                    results.append(result)
            
            return results
        
        except Exception as e:
            print(f"Error in batch classification: {str(e)}")
            return []


def classify_page_type(pdf_path: Path, page_num: int) -> Dict:
    """
    Convenience function to classify a single page.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number to analyze (1-indexed)
        
    Returns:
        Classification result dictionary
    """
    classifier = PageClassifier()
    image_bytes = classifier.pdf_page_to_image(pdf_path, page_num)
    if image_bytes:
        return classifier.classify_and_extract(image_bytes, page_num, pdf_path.name)
    return {
        'page_type': 'unknown',
        'confidence': 0.0,
        'page_number': page_num,
        'filename': pdf_path.name,
        'title_block': None,
        'error': 'Failed to convert page to image'
    }
