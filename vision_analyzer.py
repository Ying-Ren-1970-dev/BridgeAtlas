"""Vision analysis module for analyzing structural engineering CAD drawings."""
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict
import fitz  # PyMuPDF
from openai import OpenAI
from PIL import Image

import config


class VisionAnalyzer:
    """Analyzes CAD drawings using GPT-4 Vision."""
    
    def __init__(self):
        """Initialize the vision analyzer."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = "gpt-4o"  # GPT-4 with vision
    
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
            
            # Render page to pixmap with reasonable DPI
            zoom = 2  # 144 DPI (72 * 2)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            
            # Resize if needed to keep within API limits
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
    
    def analyze_drawing(self, image_bytes: bytes, page_num: int, filename: str) -> Optional[str]:
        """
        Analyze a structural engineering drawing using GPT-4 Vision.
        
        Args:
            image_bytes: Image bytes in PNG format
            page_num: Page number being analyzed
            filename: Name of the PDF file
            
        Returns:
            Textual description of the drawing or None if analysis fails
        """
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create vision prompt
            prompt = """You are analyzing a structural engineering CAD drawing. Extract and describe ALL relevant technical information using PROPER ENGINEERING TERMINOLOGY.

CRITICAL: Use standard structural engineering terms:
- **CIDH** (Cast-In-Drilled-Hole) - NOT "drilled piles" or "drilled holes filled with concrete"
- **Drilled Shaft** - Alternative term for CIDH
- **Driven Pile** - Piles driven into ground
- **Micropile** - Small diameter drilled and grouted piles
- **Augercast Pile** - Continuous flight auger piles
- Use exact pile designations shown (e.g., "24\" CIDH Type I Shaft", "72\" CIDH Type II Shaft")

STRUCTURAL COMPONENT RELATIONSHIPS:
- **Pipe Pin** = **Steel Shear Key** (transfers lateral loads between structural elements)
- **Shear Key** includes: concrete shear keys, steel shear keys, pipe pins
- Recognize functional equivalents and aliases in drawings

Extract the following:

1. **Drawing Type**: (e.g., Foundation Plan, Elevation, Section, Detail, Schedule, Bent Layout, Abutment Detail)

2. **Structural Elements**: List all elements shown (CIDH piles, driven piles, footings, columns, beams, walls, slabs, caps, shear keys, pipe pins, bearing pads, etc.)
   - **IMPORTANT**: Classify pipe pins as "Steel Shear Key (Pipe Pin)" or "Pipe Pin (Steel Shear Key)"
   - Shear keys include: concrete shear keys, steel shear keys, pipe pins

3. **Pile/Foundation Information**: 
   - **Types**: Use exact terminology (CIDH, Type I/II Shaft, driven pile, micropile, etc.)
   - **Sizes and Dimensions**: Diameters, lengths (e.g., 24", 30", 72" CIDH)
   - **Locations/Grid References**: Bent numbers, abutment numbers, station numbers, grid lines
   - **Depths and Tip Elevations**: Design tip elevation, specified tip elevation, cut-off elevation
   - **Reinforcement Details**: Bar sizes, spacing, spiral/hoop details, cage configuration
   - **Casing Information**: Steel casing thickness, permanent vs temporary, casing tip elevation

4. **Key Dimensions**: Critical measurements, spacing, elevations, clearances

5. **Annotations**: Important notes, callouts, specifications, load criteria (compression, tension, lateral, settlement)

6. **Material Specifications**: Concrete class/strength, steel grades, casing material

7. **Grid/Reference System**: Bent numbers (e.g., Bent 5, Bent 10), abutment numbers, station numbers, grid lines

8. **Connection Details**: Column-to-pile connections, cap beam details, embedment depths

Be specific and comprehensive. Use exact terms from the drawing. Include all pile schedules, tables, and detail callouts with precise designations."""

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
                max_tokens=1500,
                temperature=0.3
            )
            
            description = response.choices[0].message.content
            return description
        
        except Exception as e:
            print(f"Error analyzing drawing: {str(e)}")
            return None
    
    def analyze_pdf_page(self, pdf_path: Path, page_num: int) -> Optional[str]:
        """
        Analyze a single PDF page (convert to image and analyze).
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            
        Returns:
            Visual description of the page or None if analysis fails
        """
        # Convert page to image
        image_bytes = self.pdf_page_to_image(pdf_path, page_num)
        if not image_bytes:
            return None
        
        # Analyze the image
        description = self.analyze_drawing(image_bytes, page_num, pdf_path.name)
        return description
