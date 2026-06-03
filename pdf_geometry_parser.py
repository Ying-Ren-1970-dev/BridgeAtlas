"""
Deterministic PDF geometry parser for extracting geometric relationships from CAD drawings.
Uses pdfplumber to extract raw vector lines, bounding boxes, and text labels.
Uses shapely to compute geometric relationships (intersections, containment, etc).
No ML/Vision required - purely algorithmic geometric analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import pdfplumber
from shapely.geometry import box, LineString, Point, Polygon
from shapely.ops import unary_union
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a rectangular bounding box."""
    x0: float
    y0: float
    x1: float
    y1: float
    
    def to_shapely(self) -> box:
        """Convert to Shapely box geometry."""
        return box(self.x0, self.y0, self.x1, self.y1)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside bbox."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1
    
    def intersects_bbox(self, other: 'BoundingBox') -> bool:
        """Check if bboxes intersect."""
        return (self.x0 < other.x1 and self.x1 > other.x0 and
                self.y0 < other.y1 and self.y1 > other.y0)


@dataclass
class TextLabel:
    """Represents text extracted from a PDF page."""
    text: str
    bbox: BoundingBox
    font_size: Optional[float] = None
    is_bold: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'bbox': asdict(self.bbox),
            'font_size': self.font_size,
            'is_bold': self.is_bold
        }


@dataclass
class GeometricElement:
    """Represents a geometric element (shape, line, callout) on a page."""
    element_id: str  # Unique ID within page
    element_type: str  # "line", "rect", "circle", "path", "text_label"
    bbox: BoundingBox
    geometry: Any  # Shapely geometry object (LineString, Polygon, etc.)
    text_labels: List[TextLabel]  # Associated text labels
    properties: Dict[str, Any]  # Additional properties
    
    def to_dict(self) -> Dict:
        return {
            'element_id': self.element_id,
            'element_type': self.element_type,
            'bbox': asdict(self.bbox),
            'text_labels': [label.to_dict() for label in self.text_labels],
            'properties': self.properties
        }


@dataclass
class DetailCallout:
    """Represents an identified detail/callout on a page."""
    detail_id: str  # Format: "Page{page_num}_Detail{index}"
    page_number: int
    title: str  # Extracted from text labels
    element_type: str  # "detail", "elevation", "section", "schedule", etc.
    bbox: BoundingBox
    associated_elements: List[str]  # IDs of geometric elements that comprise this detail
    referenced_elements: List[str]  # IDs of elements this detail references/calls out
    text_content: List[str]  # All text labels associated with this detail
    
    def to_dict(self) -> Dict:
        return {
            'detail_id': self.detail_id,
            'page_number': self.page_number,
            'title': self.title,
            'element_type': self.element_type,
            'bbox': asdict(self.bbox),
            'associated_elements': self.associated_elements,
            'referenced_elements': self.referenced_elements,
            'text_content': self.text_content
        }


class PDFGeometryParser:
    """
    Deterministic parser for extracting geometric relationships from CAD PDFs.
    """
    
    # Keywords that indicate detail/section callouts
    DETAIL_KEYWORDS = [
        'detail', 'section', 'elevation', 'schedule', 'plan',
        'layout', 'diagram', 'sheet', 'view', 'enlarged'
    ]
    
    # Keywords that indicate connections/references
    CONNECTION_KEYWORDS = [
        'see', 'refer', 'shown', 'match', 'align', 'connects',
        'details', 'sections', 'notes', 'reference'
    ]
    
    def __init__(self, max_pages: Optional[int] = None):
        """
        Initialize parser.
        
        Args:
            max_pages: Maximum number of pages to process (None = all)
        """
        self.max_pages = max_pages
        self.min_text_bbox_area = 10  # Minimum area for text bboxes to consider as labels
    
    def parse_pdf(self, pdf_path: Path) -> Dict[int, Dict[str, Any]]:
        """
        Parse a PDF file for geometric elements and detail callouts.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary mapping page_num → {
                'page_number': int,
                'width': float,
                'height': float,
                'elements': List[GeometricElement.to_dict()],
                'detail_callouts': List[DetailCallout.to_dict()],
                'text_labels': List[TextLabel.to_dict()]
            }
        """
        geometry_data = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages
                
                logger.info(f"Parsing PDF geometry: {pdf_path.name}")
                logger.info(f"Total pages: {total_pages}, Processing: {pages_to_process}")
                
                for page_num in tqdm(range(pages_to_process), desc="Parsing geometry"):
                    try:
                        page = pdf.pages[page_num]
                        page_data = self._parse_page(page, page_num + 1)
                        geometry_data[page_num + 1] = page_data
                    except Exception as e:
                        logger.error(f"Error parsing page {page_num + 1}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {str(e)}")
            return {}
        
        return geometry_data
    
    def _parse_page(self, page: pdfplumber.PDF.pages, page_num: int) -> Dict[str, Any]:
        """Parse a single page for geometric elements."""
        
        # Extract raw geometric elements
        elements = self._extract_geometric_elements(page, page_num)
        
        # Extract and cluster text labels
        text_labels = self._extract_text_labels(page)
        
        # Identify detail callouts by clustering nearby elements and text
        detail_callouts = self._identify_detail_callouts(elements, text_labels, page_num)
        
        return {
            'page_number': page_num,
            'width': page.width,
            'height': page.height,
            'elements': [elem.to_dict() for elem in elements],
            'detail_callouts': [dc.to_dict() for dc in detail_callouts],
            'text_labels': [label.to_dict() for label in text_labels]
        }
    
    def _extract_geometric_elements(self, page: pdfplumber.PDF.pages, page_num: int) -> List[GeometricElement]:
        """
        Extract geometric elements (lines, rects, circles) from page.
        """
        elements = []
        element_idx = 0
        
        try:
            # Extract lines
            if hasattr(page, 'lines'):
                for line_obj in page.lines:
                    try:
                        x0, y0, x1, y1 = line_obj['x0'], line_obj['y0'], line_obj['x1'], line_obj['y1']
                        bbox = BoundingBox(
                            x0=min(x0, x1),
                            y0=min(y0, y1),
                            x1=max(x0, x1),
                            y1=max(y0, y1)
                        )
                        geometry = LineString([(x0, y0), (x1, y1)])
                        
                        elem = GeometricElement(
                            element_id=f"L{element_idx}",
                            element_type="line",
                            bbox=bbox,
                            geometry=geometry,
                            text_labels=[],
                            properties={
                                'linewidth': line_obj.get('linewidth', 1),
                                'stroking_color': str(line_obj.get('stroking_color', 'black'))
                            }
                        )
                        elements.append(elem)
                        element_idx += 1
                    except Exception as e:
                        logger.debug(f"Error parsing line: {e}")
                        continue
        except Exception as e:
            logger.debug(f"No lines extracted from page {page_num}: {e}")
        
        try:
            # Extract rectangles
            if hasattr(page, 'rects'):
                for rect_obj in page.rects:
                    try:
                        x0, y0, x1, y1 = rect_obj['x0'], rect_obj['y0'], rect_obj['x1'], rect_obj['y1']
                        bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
                        geometry = box(x0, y0, x1, y1)
                        
                        elem = GeometricElement(
                            element_id=f"R{element_idx}",
                            element_type="rect",
                            bbox=bbox,
                            geometry=geometry,
                            text_labels=[],
                            properties={
                                'linewidth': rect_obj.get('linewidth', 1),
                                'stroking_color': str(rect_obj.get('stroking_color', 'black')),
                                'non_stroking_color': str(rect_obj.get('non_stroking_color', 'transparent'))
                            }
                        )
                        elements.append(elem)
                        element_idx += 1
                    except Exception as e:
                        logger.debug(f"Error parsing rect: {e}")
                        continue
        except Exception as e:
            logger.debug(f"No rects extracted from page {page_num}: {e}")
        
        try:
            # Extract curves (circles, arcs, etc.)
            if hasattr(page, 'curves'):
                for curve_obj in page.curves:
                    try:
                        x0, y0, x1, y1 = curve_obj['x0'], curve_obj['y0'], curve_obj['x1'], curve_obj['y1']
                        bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
                        # Approximate curve as polygon
                        geometry = box(x0, y0, x1, y1)  # Simplified
                        
                        elem = GeometricElement(
                            element_id=f"C{element_idx}",
                            element_type="curve",
                            bbox=bbox,
                            geometry=geometry,
                            text_labels=[],
                            properties={'type': 'curve'}
                        )
                        elements.append(elem)
                        element_idx += 1
                    except Exception as e:
                        logger.debug(f"Error parsing curve: {e}")
                        continue
        except Exception as e:
            logger.debug(f"No curves extracted from page {page_num}: {e}")
        
        return elements
    
    def _extract_text_labels(self, page: pdfplumber.PDF.pages) -> List[TextLabel]:
        """
        Extract text labels and their bounding boxes from page.
        """
        text_labels = []
        
        try:
            for char_obj in page.chars:
                try:
                    text = char_obj.get('text', '')
                    if not text or text.isspace():
                        continue
                    
                    bbox = BoundingBox(
                        x0=char_obj['x0'],
                        y0=char_obj['y0'],
                        x1=char_obj['x1'],
                        y1=char_obj['y1']
                    )
                    
                    label = TextLabel(
                        text=text,
                        bbox=bbox,
                        font_size=char_obj.get('size'),
                        is_bold='Bold' in str(char_obj.get('fontname', ''))
                    )
                    text_labels.append(label)
                except Exception as e:
                    logger.debug(f"Error extracting text label: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Error extracting text labels: {e}")
        
        # Also try extracting words (grouped text)
        try:
            for word in page.extract_words():
                try:
                    text = word.get('text', '')
                    if not text or len(text) < 2:
                        continue
                    
                    bbox = BoundingBox(
                        x0=word['x0'],
                        y0=word['y0'],
                        x1=word['x1'],
                        y1=word['y1']
                    )
                    
                    label = TextLabel(
                        text=text,
                        bbox=bbox,
                        font_size=word.get('size')
                    )
                    text_labels.append(label)
                except Exception as e:
                    logger.debug(f"Error extracting word: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Error extracting words: {e}")
        
        return text_labels
    
    def _identify_detail_callouts(
        self,
        elements: List[GeometricElement],
        text_labels: List[TextLabel],
        page_num: int
    ) -> List[DetailCallout]:
        """
        Identify detail callouts by clustering elements and matching with text labels.
        
        A detail callout is typically:
        - A circle or rectangle with a number/label inside or nearby
        - Associated text that indicates it's a detail reference (e.g., "Detail A", "Section 2-2")
        """
        callouts = []
        callout_idx = 0
        
        # Look for text labels that match detail keywords
        for text_label in text_labels:
            text_lower = text_label.text.lower()
            
            # Check if this text indicates a detail/section/elevation
            is_detail_indicator = any(
                keyword in text_lower for keyword in self.DETAIL_KEYWORDS
            )
            
            if not is_detail_indicator:
                continue
            
            # Find nearby elements (within ~50 points)
            nearby_elements = []
            associated_bbox = text_label.bbox
            
            for elem in elements:
                if self._bboxes_are_close(text_label.bbox, elem.bbox, threshold=100):
                    nearby_elements.append(elem.element_id)
                    # Expand bbox to include element
                    associated_bbox = self._union_bbox(associated_bbox, elem.bbox)
            
            # Create detail callout
            callout = DetailCallout(
                detail_id=f"Page{page_num}_Detail{callout_idx}",
                page_number=page_num,
                title=text_label.text,
                element_type=self._classify_element_type(text_label.text),
                bbox=associated_bbox,
                associated_elements=nearby_elements,
                referenced_elements=[],
                text_content=[text_label.text]
            )
            callouts.append(callout)
            callout_idx += 1
        
        return callouts
    
    def _bboxes_are_close(self, bbox1: BoundingBox, bbox2: BoundingBox, threshold: float = 100) -> bool:
        """Check if two bboxes are within threshold distance."""
        # Calculate center-to-center distance
        c1_x, c1_y = (bbox1.x0 + bbox1.x1) / 2, (bbox1.y0 + bbox1.y1) / 2
        c2_x, c2_y = (bbox2.x0 + bbox2.x1) / 2, (bbox2.y0 + bbox2.y1) / 2
        
        distance = ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5
        return distance <= threshold
    
    def _union_bbox(self, bbox1: BoundingBox, bbox2: BoundingBox) -> BoundingBox:
        """Compute union of two bounding boxes."""
        return BoundingBox(
            x0=min(bbox1.x0, bbox2.x0),
            y0=min(bbox1.y0, bbox2.y0),
            x1=max(bbox1.x1, bbox2.x1),
            y1=max(bbox1.y1, bbox2.y1)
        )
    
    def _classify_element_type(self, text: str) -> str:
        """Classify element type based on text content."""
        text_lower = text.lower()
        
        if 'section' in text_lower:
            return 'section'
        elif 'elevation' in text_lower:
            return 'elevation'
        elif 'schedule' in text_lower:
            return 'schedule'
        elif 'plan' in text_lower:
            return 'plan'
        elif 'detail' in text_lower:
            return 'detail'
        elif 'layout' in text_lower:
            return 'layout'
        elif 'diagram' in text_lower:
            return 'diagram'
        else:
            return 'detail'  # Default
    
    def save_geometry_data(self, geometry_data: Dict, output_path: Path) -> None:
        """Save parsed geometry data to JSON file."""
        try:
            # Convert to JSON-serializable format
            json_data = {}
            for page_num, page_data in geometry_data.items():
                json_data[str(page_num)] = page_data
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Saved geometry data to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save geometry data: {e}")
