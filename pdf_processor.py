"""PDF processing module for extracting text and metadata from structural plans."""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import PyPDF2
import pdfplumber
from tqdm import tqdm

import config
from vision_analyzer import VisionAnalyzer


class PDFProcessor:
    """Handles PDF text extraction and preprocessing."""
    
    def __init__(self, use_vision=True):
        self.max_pages = config.MAX_PAGES_PER_PDF
        self.use_vision = use_vision
        if use_vision:
            self.vision_analyzer = VisionAnalyzer()
        else:
            self.vision_analyzer = None
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[Dict[int, str], int]:
        """
        Extract text and analyze CAD drawings from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (dictionary mapping page numbers to text+vision content, actual total page count)
        """
        pages_text = {}
        actual_total_pages = 0
        
        try:
            # Try pdfplumber first (better for structured documents)
            with pdfplumber.open(pdf_path) as pdf:
                actual_total_pages = len(pdf.pages)
                total_pages = min(len(pdf.pages), self.max_pages)
                
                print(f"\nProcessing {pdf_path.name}...")
                print(f"Total pages: {actual_total_pages}, Processing: {total_pages}")
                
                for page_num in tqdm(range(total_pages), desc="Extracting"):
                    try:
                        page = pdf.pages[page_num]
                        text = page.extract_text() or ""
                        text_length = len(text.strip())
                        
                        # Decide if we need vision analysis
                        needs_vision = False
                        if text_length == 0:
                            # No text - definitely a drawing
                            needs_vision = True
                        elif text_length < 300:
                            # Minimal text - likely a CAD drawing with labels
                            needs_vision = True
                        
                        if needs_vision and self.use_vision and self.vision_analyzer:
                            print(f"\n  Page {page_num + 1}: Analyzing CAD drawing (text: {text_length} chars)...")
                            vision_description = self.vision_analyzer.analyze_pdf_page(pdf_path, page_num + 1)
                            
                            if vision_description:
                                if text.strip():
                                    # Combine text and vision
                                    combined = f"[TEXT CONTENT]\n{text}\n\n[DRAWING ANALYSIS]\n{vision_description}"
                                    pages_text[page_num + 1] = combined
                                else:
                                    # Vision only
                                    pages_text[page_num + 1] = f"[DRAWING ANALYSIS]\n{vision_description}"
                            elif text.strip():
                                # Vision failed, use text only
                                pages_text[page_num + 1] = text
                        elif text.strip():
                            # Sufficient text, no vision needed
                            pages_text[page_num + 1] = text
                            
                    except Exception as e:
                        print(f"Error extracting page {page_num + 1} from {pdf_path.name}: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path.name}, trying PyPDF2: {str(e)}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    actual_total_pages = len(pdf_reader.pages)
                    total_pages = min(len(pdf_reader.pages), self.max_pages)
                    
                    print(f"\nProcessing {pdf_path.name} with PyPDF2...")
                    print(f"Total pages: {actual_total_pages}, Processing: {total_pages}")
                    
                    for page_num in tqdm(range(total_pages), desc="Extracting"):
                        try:
                            page = pdf_reader.pages[page_num]
                            text = page.extract_text() or ""
                            text_length = len(text.strip())
                            
                            needs_vision = text_length < 300
                            
                            if needs_vision and self.use_vision and self.vision_analyzer:
                                print(f"\n  Page {page_num + 1}: Analyzing CAD drawing (text: {text_length} chars)...")
                                vision_description = self.vision_analyzer.analyze_pdf_page(pdf_path, page_num + 1)
                                
                                if vision_description:
                                    if text.strip():
                                        combined = f"[TEXT CONTENT]\n{text}\n\n[DRAWING ANALYSIS]\n{vision_description}"
                                        pages_text[page_num + 1] = combined
                                    else:
                                        pages_text[page_num + 1] = f"[DRAWING ANALYSIS]\n{vision_description}"
                                elif text.strip():
                                    pages_text[page_num + 1] = text
                            elif text.strip():
                                pages_text[page_num + 1] = text
                        except Exception as e:
                            print(f"Error extracting page {page_num + 1}: {str(e)}")
                            continue
                            
            except Exception as e:
                print(f"Failed to process {pdf_path.name}: {str(e)}")
        
        return pages_text, actual_total_pages
    
    def extract_metadata_from_text(self, text: str, filename: str) -> Dict[str, Optional[str]]:
        """
        Extract metadata from PDF text using pattern matching.
        
        Args:
            text: Extracted text from PDF
            filename: Name of the PDF file
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            'project_name': None,
            'phase': None,
            'engineer_of_record': None,
            'date': None,
        }
        
        # Extract project name from filename if not found in text
        metadata['project_name'] = self._extract_project_name(filename)
        
        # Extract phase
        metadata['phase'] = self._extract_phase(text, filename)
        
        # Extract engineer/firm
        metadata['engineer_of_record'] = self._extract_engineer(text)
        
        # Extract date
        metadata['date'] = self._extract_date(text, filename)
        
        return metadata
    
    def _extract_project_name(self, filename: str) -> str:
        """Extract project name from filename."""
        # Remove file extension
        name = Path(filename).stem
        
        # Clean up common suffixes
        name = re.sub(r'[-_](Plan|Plans|Set|Vol|Volume|Final|Submittal).*$', '', name, flags=re.IGNORECASE)
        
        return name.strip()
    
    def _extract_phase(self, text: str, filename: str) -> Optional[str]:
        """Extract project phase from text or filename."""
        combined_text = f"{filename} {text[:2000]}"  # Check filename and first 2000 chars
        
        # Check for phase patterns
        phase_patterns = {
            'IFB': r'\bIFB\b|Invitation\s+for\s+Bid',
            '100% Final': r'100%.*(?:Final|Submittal)|Final.*100%',
            '90% Design': r'90%.*Design|Design.*90%',
            '60% Design': r'60%.*Design|Design.*60%',
            '30% Design': r'30%.*Design|Design.*30%',
            'Preliminary': r'\bPreliminary\b',
            'Conceptual': r'\bConceptual\b',
            'As-Built': r'As[-\s]Built|As[-\s]Constructed',
            'Markup': r'\bMarkup\b',
        }
        
        for phase, pattern in phase_patterns.items():
            if re.search(pattern, combined_text, re.IGNORECASE):
                return phase
        
        return None
    
    def _extract_engineer(self, text: str) -> Optional[str]:
        """Extract engineer of record from text."""
        # Look for common patterns
        patterns = [
            r'Engineer\s+of\s+Record[:\s]+([^\n]+)',
            r'Structural\s+Engineer[:\s]+([^\n]+)',
            r'Prepared\s+by[:\s]+([^\n]+)',
            r'Designed\s+by[:\s]+([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                engineer = match.group(1).strip()
                # Clean up
                engineer = re.sub(r'[,;].*$', '', engineer)
                if len(engineer) > 5 and len(engineer) < 100:
                    return engineer
        
        return None
    
    def _extract_date(self, text: str, filename: str) -> Optional[str]:
        """Extract date from text or filename."""
        combined_text = f"{filename} {text[:2000]}"
        
        # Date patterns
        patterns = [
            r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b',  # YYYY-MM-DD or YYYY/MM/DD
            r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b',  # MM-DD-YYYY or MM/DD/YYYY
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2})\b',  # M-D-YY or MM/DD/YY
            r'\b([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b',  # Month DD, YYYY
        ]
        
        for pattern in patterns:
            match = re.search(pattern, combined_text)
            if match:
                return match.group(1)
        
        return None
    
    def _is_section_aware_project(self, file_name: Optional[str]) -> bool:
        """Return True when file matches configured project patterns for section-aware chunking."""
        if not file_name:
            return False
        patterns = config.SECTION_AWARE_CHUNKING_PROJECT_PATTERNS
        if any(pattern in {"all", "*"} for pattern in patterns):
            return True
        file_name_lower = file_name.lower()
        return any(pattern in file_name_lower for pattern in patterns)

    def _split_structural_sections(self, text: str) -> List[str]:
        """Split vision-rich content into engineering sections while preserving semantics."""
        if not text:
            return []

        normalized = text.replace("\r\n", "\n")

        # Prefer numbered markdown sections emitted by vision prompt.
        section_markers = [
            r"\n(?=\d+\.\s+\*\*[^*]+\*\*:)" ,
            r"\n(?=\d+\.\s+[A-Za-z][^:\n]{2,80}:)",
            r"\n(?=\*\*[A-Za-z][^*]{2,80}\*\*:)"
        ]

        sections = [normalized]
        for marker in section_markers:
            next_sections = []
            for section in sections:
                parts = re.split(marker, section)
                if parts:
                    next_sections.extend(parts)
            sections = next_sections

        cleaned = []
        for section in sections:
            section_text = section.strip()
            if len(section_text) >= 80:
                cleaned.append(section_text)

        return cleaned

    def chunk_text(self, text: str, page_num: int, file_name: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Text to chunk
            page_num: Page number this text came from
            file_name: Optional source file name for project-specific chunking strategy
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        chunks = []
        chunk_size = config.CHUNK_SIZE
        overlap = config.CHUNK_OVERLAP

        # Keep full-page chunk + section-aware chunks to preserve drawing semantics.
        if self._is_section_aware_project(file_name):
            chunk_id = 0
            page_text = text.strip()
            if page_text:
                chunks.append({
                    'text': page_text,
                    'page': page_num,
                    'chunk_id': chunk_id,
                })
                chunk_id += 1

            for section in self._split_structural_sections(page_text):
                chunks.append({
                    'text': section,
                    'page': page_num,
                    'chunk_id': chunk_id,
                })
                chunk_id += 1

            if chunks:
                return chunks
        
        # Simple character-based chunking
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'page': page_num,
                    'chunk_id': chunk_id,
                })
                chunk_id += 1
            
            start = end - overlap
        
        return chunks
    
    def process_pdf(self, pdf_path: Path) -> Dict:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted data and metadata
        """
        print(f"\nProcessing: {pdf_path.name}")
        
        # Extract text by page
        pages_text, actual_total_pages = self.extract_text_from_pdf(pdf_path)
        
        if not pages_text:
            print(f"Warning: No text extracted from {pdf_path.name}")
            return None
        
        # Extract metadata from first few pages
        combined_text = ' '.join([pages_text.get(i, '') for i in range(1, min(6, len(pages_text) + 1))])
        metadata = self.extract_metadata_from_text(combined_text, pdf_path.name)
        
        # Create chunks
        all_chunks = []
        for page_num, page_text in pages_text.items():
            chunks = self.chunk_text(page_text, page_num, pdf_path.name)
            all_chunks.extend(chunks)
        
        return {
            'file_path': str(pdf_path),
            'file_name': pdf_path.name,
            'metadata': metadata,
            'pages_text': pages_text,
            'chunks': all_chunks,
            'total_pages': actual_total_pages,
            'pages_with_text': len(pages_text),
        }
    
    def _is_excluded_pdf(self, pdf_path: Path) -> bool:
        """Return True if the PDF path is excluded by configured folder patterns."""
        normalized_path = str(pdf_path).replace('\\', '/').lower()
        return any(pattern in normalized_path for pattern in config.PROJECTS_FOLDER_EXCLUDE)

    def process_all_pdfs(self, projects_folder: Path) -> List[Dict]:
        """
        Process all PDFs in the projects folder.
        
        Args:
            projects_folder: Root folder containing projects
            
        Returns:
            List of processed PDF data
        """
        pdf_files = [
            pdf_path
            for pdf_path in projects_folder.rglob('*.pdf')
            if not self._is_excluded_pdf(pdf_path)
        ]
        excluded_count = len(list(projects_folder.rglob('*.pdf'))) - len(pdf_files)
        print(f"\nFound {len(pdf_files)} PDF files")
        if excluded_count:
            print(f"Excluded {excluded_count} PDF files from ignored folders: {', '.join(config.PROJECTS_FOLDER_EXCLUDE)}")
        
        processed_data = []
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            result = self.process_pdf(pdf_path)
            if result:
                processed_data.append(result)
        
        return processed_data
