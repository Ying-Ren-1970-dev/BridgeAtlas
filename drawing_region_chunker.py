"""Detect drawing-region chunks with vision and assemble searchable chunk text."""
from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fitz
import openai
from tqdm import tqdm

import config
from cad_drawing_knowledge import CadDrawingKnowledge
from drawing_chunk_examples import CACHE_DIR, TRAINING_EXAMPLES
from metadata_manager import MetadataManager
from storage_adapter import storage

CHUNK_DETECTION_PROMPT = """Analyze this structural CAD plan sheet and identify each separate drawing region that should become its own knowledge-base chunk.

Rules learned from engineer-annotated training sheets:
- One chunk per labeled drawing (title + scale + drawing body).
- Chunk boundaries are approximate; leader-line text belongs where the arrowhead points.
- Common view types: plan, elevation, layout, detail, section.
- View is rare (only when labeled VIEW A-A).
- Also use label-derived types: diagram, as_built, schedule.
- Imprecise labels like TYPICAL SECTION or TYPICAL DETAILS are valid.
- Notes, legend, and title block are NOT drawing chunks.
- A large ELEVATION is ONE chunk even if it contains sub-areas (END POST, INTERMEDIATE PANELS, etc.).
- Layout sheets include PLAN STAGE N, ELEVATION STAGE N, and related SECTION A-A / SECTION B-B chunks.
- BENT CAP CAMBER DIAGRAM is its own diagram chunk on bent detail sheets.

Training examples (file / page / expected chunks):
{training_summary}

Return JSON only:
{{
  "sheet_title": "exact sheet title if visible",
  "drawing_chunks": [
    {{
      "label": "exact visible title e.g. SECTION A-A, DETAIL 1, BENT CAP CAMBER DIAGRAM",
      "view_type": "plan | elevation | layout | detail | section | view | diagram | as_built | other",
      "is_precise": true,
      "scale": "scale if shown or null",
      "description": "technical description of what this drawing shows",
      "leader_text": ["dimension or note text whose leader arrowhead points into this drawing"],
      "cross_references": ["DETAIL 5", "SECTION B-B", "see ABUTMENT DETAILS No. 1"]
    }}
  ]
}}
"""


@dataclass
class DrawingRegionSpec:
    label: str
    view_type: str = ""
    is_precise: bool = True
    scale: str = ""
    description: str = ""
    leader_text: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class DrawingRegionChunker:
    """Vision-based drawing region detection and chunk text assembly."""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = "gpt-4o"
        self._pdf_handle = None
        self._pdf_handle_path: Optional[Path] = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_pdf(self, pdf_path: Path):
        """Reuse one pdfplumber handle per PDF (avoids reopening on every region)."""
        resolved = pdf_path.resolve()
        if self._pdf_handle_path != resolved:
            self.close_pdf()
            import pdfplumber

            self._pdf_handle = pdfplumber.open(resolved)
            self._pdf_handle_path = resolved
        return self._pdf_handle

    def close_pdf(self) -> None:
        if self._pdf_handle is not None:
            try:
                self._pdf_handle.close()
            except Exception:
                pass
        self._pdf_handle = None
        self._pdf_handle_path = None

    @staticmethod
    def _training_summary() -> str:
        lines = []
        for example in TRAINING_EXAMPLES:
            labels = ", ".join(chunk["label"] for chunk in example["expected_chunks"])
            lines.append(
                f"- {example['file_name']} p{example['page']} "
                f"({example['sheet_title']}): {labels}"
            )
        return "\n".join(lines)

    def _cache_path(self, file_name: str) -> Path:
        stem = Path(file_name).stem
        return CACHE_DIR / f"{stem}.json"

    def _load_cache(self, file_name: str) -> Dict[str, List[Dict]]:
        path = self._cache_path(file_name)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_cache(self, file_name: str, cache: Dict[str, List[Dict]]) -> None:
        path = self._cache_path(file_name)
        path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    def _resolve_pdf_path(self, file_name: str) -> Path:
        pdf_path = storage._resolve_local_pdf_path(file_name)
        if pdf_path and pdf_path.exists():
            return pdf_path
        raise FileNotFoundError(f"PDF not found for {file_name}")

    def _call_chunk_detection_api(self, base64_image: str, prompt: str) -> Dict:
        """Call vision API with retry when JSON is truncated."""
        last_error: Optional[Exception] = None
        for attempt, max_tokens in enumerate((3500, 5000), start=1):
            try:
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
                                        "detail": "high",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                print(
                    f"  Warning: chunk detection JSON parse failed "
                    f"(attempt {attempt}); retrying with larger token budget"
                )
            except Exception as exc:
                last_error = exc
                break

        print(f"  Warning: chunk detection failed, using fallback: {last_error}")
        return {"drawing_chunks": []}

    def _page_to_image(self, pdf_path: Path, page_num: int) -> bytes:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        data = pix.tobytes("png")
        doc.close()
        return data

    def detect_page_regions(
        self,
        file_name: str,
        page_num: int,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> List[DrawingRegionSpec]:
        """Detect drawing-region chunks on one plan sheet page."""
        cache = self._load_cache(file_name)
        cache_key = str(page_num)
        if use_cache and not force_refresh and cache_key in cache:
            return [DrawingRegionSpec(**item) for item in cache[cache_key]]

        pdf_path = self._resolve_pdf_path(file_name)
        image_bytes = self._page_to_image(pdf_path, page_num)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        prompt = CHUNK_DETECTION_PROMPT.format(
            training_summary=self._training_summary()
        )

        payload = self._call_chunk_detection_api(base64_image, prompt)
        regions: List[DrawingRegionSpec] = []
        for item in payload.get("drawing_chunks") or []:
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            classification = CadDrawingKnowledge.classify_label(label)
            regions.append(
                DrawingRegionSpec(
                    label=label,
                    view_type=str(item.get("view_type") or classification.view_type),
                    is_precise=bool(item.get("is_precise", classification.is_precise)),
                    scale=str(item.get("scale") or ""),
                    description=str(item.get("description") or ""),
                    leader_text=[
                        str(value).strip()
                        for value in (item.get("leader_text") or [])
                        if str(value).strip()
                    ],
                    cross_references=[
                        str(value).strip()
                        for value in (item.get("cross_references") or [])
                        if str(value).strip()
                    ],
                )
            )

        cache[cache_key] = [region.to_dict() for region in regions]
        self._save_cache(file_name, cache)
        return regions

    @staticmethod
    def _normalize_label(label: str) -> str:
        return re.sub(r"\s+", " ", label).strip().upper()

    @staticmethod
    def _label_pattern(label: str) -> str:
        return re.escape(label).replace(r"\ ", r"\s+")

    @staticmethod
    def _sanitize_page_source_text(page_text: str) -> str:
        """Drop recycled chunk/vision blocks so label windows stay local."""
        if not page_text:
            return ""
        cleaned = page_text
        for marker in (
            "[DRAWING REGION CHUNK]",
            "[DRAWING ANALYSIS]",
            "[ENRICHED METADATA]",
            "[CAD DRAWING KNOWLEDGE]",
            "[LABEL CONTEXT]",
        ):
            cleaned = cleaned.replace(marker, "\n")
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    @staticmethod
    def _extract_raw_pdf_page_text(pdf_path: Path, page_num: int) -> str:
        try:
            import pdfplumber
        except ImportError:
            return ""

        try:
            pdf = self._get_pdf(pdf_path)
            if page_num < 1 or page_num > len(pdf.pages):
                return ""
            return pdf.pages[page_num - 1].extract_text() or ""
        except Exception:
            return ""

    def _extract_bounded_label_context(
        self,
        page_text: str,
        label: str,
        sibling_labels: Sequence[str],
        max_chars: int = 900,
    ) -> str:
        """Extract OCR text bounded by sibling drawing labels on the same page."""
        page_text = self._sanitize_page_source_text(page_text)
        if not page_text or not label:
            return ""

        pattern = self._label_pattern(label)
        match = re.search(pattern, page_text, re.IGNORECASE)
        if not match:
            return ""

        start, end = match.start(), match.end()
        left_bound = 0
        right_bound = len(page_text)
        target = self._normalize_label(label)

        for other in sibling_labels:
            if self._normalize_label(other) == target:
                continue
            other_pattern = self._label_pattern(other)
            for other_match in re.finditer(other_pattern, page_text, re.IGNORECASE):
                if other_match.end() <= start and other_match.end() > left_bound:
                    left_bound = other_match.end()
                if other_match.start() >= end and other_match.start() < right_bound:
                    right_bound = other_match.start()

        snippet = re.sub(r"\s+", " ", page_text[left_bound:right_bound]).strip()
        if len(snippet) <= max_chars:
            return snippet

        label_offset = snippet.upper().find(self._normalize_label(label).split()[0])
        if label_offset < 0:
            return snippet[:max_chars]
        half = max_chars // 2
        slice_start = max(0, label_offset - half)
        return snippet[slice_start : slice_start + max_chars]

    @staticmethod
    def _find_label_word_span(words: List[dict], label: str) -> Optional[Tuple[int, int]]:
        tokens = [token for token in re.split(r"\s+", label.strip()) if token]
        if not tokens or len(tokens) > len(words):
            return None

        upper_tokens = [token.upper() for token in tokens]
        for index in range(len(words) - len(tokens) + 1):
            window = [words[index + offset]["text"].upper() for offset in range(len(tokens))]
            if window == upper_tokens:
                return index, index + len(tokens)
        return None

    @staticmethod
    def _format_spatial_words(words: List[dict], max_chars: int = 2000) -> str:
        lines: List[str] = []
        current_line: List[dict] = []
        last_top: Optional[float] = None

        for word in sorted(words, key=lambda item: (item.get("top", 0), item.get("x0", 0))):
            top = round(word.get("top", 0), 0)
            if last_top is not None and abs(top - last_top) > 4 and current_line:
                lines.append(" ".join(item["text"] for item in current_line))
                current_line = []
            current_line.append(word)
            last_top = top

        if current_line:
            lines.append(" ".join(item["text"] for item in current_line))

        return "\n".join(lines)[:max_chars]

    @staticmethod
    def _word_bbox(words: List[dict]) -> Tuple[float, float, float, float]:
        return (
            min(item["x0"] for item in words),
            min(item["top"] for item in words),
            max(item["x1"] for item in words),
            max(item["bottom"] for item in words),
        )

    @staticmethod
    def _expand_bbox_with_midpoint_clips(
        bbox: Tuple[float, float, float, float],
        center: Tuple[float, float],
        peer_boxes: List[Tuple[Tuple[float, float], Tuple[float, float, float, float]]],
        page_width: float,
        page_height: float,
        margin: float = 140.0,
    ) -> Tuple[float, float, float, float]:
        x0, y0, x1, y1 = bbox
        cx, cy = center
        x0 -= margin
        y0 -= margin
        x1 += margin
        y1 += margin

        for peer_center, peer_bbox in peer_boxes:
            pcx, pcy = peer_center
            if abs(pcx - cx) >= abs(pcy - cy):
                midpoint = (cx + pcx) / 2
                if pcx > cx:
                    x1 = min(x1, midpoint)
                else:
                    x0 = max(x0, midpoint)
            else:
                midpoint = (cy + pcy) / 2
                if pcy > cy:
                    y1 = min(y1, midpoint)
                else:
                    y0 = max(y0, midpoint)

        return (
            max(0.0, x0),
            max(0.0, y0),
            min(page_width, x1),
            min(page_height, y1),
        )

    def _scrub_sibling_labels(
        self,
        text: str,
        label: str,
        sibling_labels: Sequence[str],
    ) -> str:
        cleaned = text
        own = self._normalize_label(label)
        for other in sibling_labels:
            if self._normalize_label(other) == own:
                continue
            if own and self._normalize_label(other) in own:
                continue
            pattern = rf"(?i)\b{self._label_pattern(other)}\b"
            cleaned = re.sub(pattern, " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _extract_spatial_region_text(
        self,
        pdf_path: Path,
        page_num: int,
        label: str,
        sibling_labels: Sequence[str],
    ) -> str:
        """Collect PDF words inside a clipped bounding box around one drawing label."""
        try:
            import pdfplumber
        except ImportError:
            return ""

        try:
            pdf = self._get_pdf(pdf_path)
            if page_num < 1 or page_num > len(pdf.pages):
                return ""
            page = pdf.pages[page_num - 1]
            words = page.extract_words() or []
            page_width = float(page.width)
            page_height = float(page.height)
        except Exception:
            return ""

        if not words:
            return ""

        words = sorted(words, key=lambda item: (round(item.get("top", 0), 1), item.get("x0", 0)))
        label_boxes: Dict[str, Tuple[Tuple[float, float], Tuple[float, float, float, float]]] = {}
        for region_label in dict.fromkeys([label, *sibling_labels]):
            span = self._find_label_word_span(words, region_label)
            if span is None:
                continue
            matched = words[span[0] : span[1]]
            bbox = self._word_bbox(matched)
            center = (
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2,
            )
            label_boxes[region_label] = (center, bbox)

        if label not in label_boxes:
            return ""

        center, bbox = label_boxes[label]
        peer_boxes = [
            (label_boxes[peer][0], label_boxes[peer][1])
            for peer in label_boxes
            if peer != label
        ]
        region = self._expand_bbox_with_midpoint_clips(
            bbox=bbox,
            center=center,
            peer_boxes=peer_boxes,
            page_width=page_width,
            page_height=page_height,
        )

        owned_words: List[dict] = []
        for word in words:
            word_x = (word["x0"] + word["x1"]) / 2
            word_y = (word["top"] + word["bottom"]) / 2
            if region[0] <= word_x <= region[2] and region[1] <= word_y <= region[3]:
                owned_words.append(word)

        if not owned_words:
            return ""
        return self._format_spatial_words(owned_words)

    def _extract_region_context(
        self,
        page_text: str,
        label: str,
        sibling_labels: Sequence[str],
        pdf_path: Optional[Path] = None,
        page_num: int = 0,
    ) -> str:
        peers = list(sibling_labels)
        if pdf_path and page_num:
            spatial_text = self._extract_spatial_region_text(
                pdf_path=pdf_path,
                page_num=page_num,
                label=label,
                sibling_labels=peers,
            )
            if spatial_text:
                return self._scrub_sibling_labels(spatial_text, label, peers)

            raw_page_text = self._extract_raw_pdf_page_text(pdf_path, page_num)
            if raw_page_text:
                bounded = self._extract_bounded_label_context(raw_page_text, label, peers)
                scrubbed = self._scrub_sibling_labels(bounded, label, peers)
                if scrubbed:
                    return scrubbed

        bounded = self._extract_bounded_label_context(page_text, label, peers)
        return self._scrub_sibling_labels(bounded, label, peers)

    def assemble_chunk_text(
        self,
        region: DrawingRegionSpec,
        page_text: str,
        page_num: int,
        file_name: str,
        sibling_labels: Optional[Sequence[str]] = None,
        pdf_path: Optional[Path] = None,
    ) -> str:
        """Build searchable text for one drawing-region chunk."""
        peers = [
            peer
            for peer in (sibling_labels or [])
            if self._normalize_label(peer) != self._normalize_label(region.label)
        ]
        peer_keys = {self._normalize_label(peer) for peer in peers}

        lines = [
            f"[DRAWING REGION CHUNK]",
            f"Sheet: {file_name} page {page_num}",
            f"Drawing Label: {region.label}",
            f"Drawing View: {region.view_type}",
        ]
        if region.scale:
            lines.append(f"Scale: {region.scale}")
        if region.description:
            lines.append(f"Description: {region.description}")
        if region.leader_text:
            lines.append("Leader Text:")
            lines.extend(f"- {item}" for item in region.leader_text[:12])
        if region.cross_references:
            filtered_refs = []
            for item in region.cross_references[:12]:
                normalized_item = self._normalize_label(item)
                if any(peer in normalized_item for peer in peer_keys):
                    continue
                filtered_refs.append(item)
            if filtered_refs:
                lines.append("Cross References:")
                lines.extend(f"- {item}" for item in filtered_refs)

        ocr_context = self._extract_region_context(
            page_text=page_text,
            label=region.label,
            sibling_labels=peers,
            pdf_path=pdf_path,
            page_num=page_num,
        )
        if ocr_context:
            lines.append("[LABEL CONTEXT]")
            lines.append(ocr_context)

        return "\n".join(lines)

    @staticmethod
    def is_plan_page(page_text: str, page_enriched: Optional[Dict] = None) -> bool:
        if page_enriched:
            if page_enriched.get("page_type") == "report":
                return False
            if page_enriched.get("page_type") == "plan":
                return True
        lowered = (page_text or "").lower()
        if "[drawing analysis]" in lowered:
            return True
        drawing_signals = (
            "section ",
            "detail ",
            "elevation",
            "plan",
            "layout",
            "diagram",
            "typical section",
            "typical details",
        )
        return any(signal in lowered for signal in drawing_signals)

    def build_page_chunks(
        self,
        file_name: str,
        page_num: int,
        page_text: str,
        page_enriched: Optional[Dict] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> List[Dict]:
        """Return drawing-region chunks for a page, or a single fallback chunk."""
        if not self.is_plan_page(page_text, page_enriched):
            text = page_text.strip()
            if not text:
                return []
            return [{"text": text, "page": page_num, "chunk_id": 0}]

        regions = self.detect_page_regions(
            file_name=file_name,
            page_num=page_num,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )
        if not regions:
            text = page_text.strip()
            return [{"text": text, "page": page_num, "chunk_id": 0}] if text else []

        pdf_path: Optional[Path] = None
        try:
            pdf_path = self._resolve_pdf_path(file_name)
        except FileNotFoundError:
            pdf_path = None

        region_labels = [region.label for region in regions]
        chunks: List[Dict] = []
        for chunk_id, region in enumerate(regions):
            chunk_text = self.assemble_chunk_text(
                region,
                page_text,
                page_num,
                file_name,
                sibling_labels=region_labels,
                pdf_path=pdf_path,
            )
            classification = CadDrawingKnowledge.classify_label(region.label)
            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "drawing_label": region.label,
                    "drawing_view_type": region.view_type or classification.view_type,
                    "drawing_precise": region.is_precise,
                }
            )
        return chunks


class DrawingChunkApplicator:
    """Apply trained drawing-region chunking across all indexed projects."""

    def __init__(self, chunker: Optional[DrawingRegionChunker] = None):
        self.chunker = chunker or DrawingRegionChunker()

    @staticmethod
    def _load_existing_chunks(collection, file_name: str) -> List[Dict]:
        results = collection.get(
            where={"file_name": file_name},
            include=["metadatas", "documents"],
        )
        chunks: List[Dict] = []
        for metadata, document in zip(results["metadatas"], results["documents"]):
            chunks.append(
                {
                    "text": document,
                    "page": int(metadata.get("page") or 0),
                    "chunk_id": int(metadata.get("chunk_id") or 0),
                    "drawing_label": metadata.get("cad_drawing_label"),
                    "drawing_view_type": metadata.get("cad_drawing_view"),
                }
            )
        return chunks

    @staticmethod
    def _load_page_text_from_index(collection, file_name: str) -> Dict[int, str]:
        results = collection.get(
            where={"file_name": file_name},
            include=["metadatas", "documents"],
        )
        by_page: Dict[int, List[tuple]] = {}
        for metadata, document in zip(results["metadatas"], results["documents"]):
            page = int(metadata.get("page") or 0)
            chunk_id = int(metadata.get("chunk_id") or 0)
            by_page.setdefault(page, []).append((chunk_id, document))

        page_text: Dict[int, str] = {}
        for page, chunks in by_page.items():
            chunks.sort(key=lambda item: item[0])
            # Prefer the longest chunk as the richest page representation.
            best = max(chunks, key=lambda item: len(item[1]))
            page_text[page] = best[1]
        return page_text

    def _load_raw_pdf_pages(
        self,
        file_name: str,
        page_nums: Optional[Sequence[int]] = None,
    ) -> Dict[int, str]:
        try:
            pdf_path = storage._resolve_local_pdf_path(file_name)
        except Exception:
            return {}
        if not pdf_path or not pdf_path.exists():
            return {}

        pages: Dict[int, str] = {}
        try:
            doc = fitz.open(pdf_path)
            targets = (
                sorted({int(page) for page in page_nums})
                if page_nums
                else range(1, doc.page_count + 1)
            )
            for page_num in targets:
                if page_num < 1 or page_num > doc.page_count:
                    continue
                pages[page_num] = doc[page_num - 1].get_text() or ""
            doc.close()
        except Exception:
            return {}
        return pages

    def _load_pages_for_chunking(
        self,
        collection,
        file_name: str,
        page_nums: Sequence[int],
    ) -> Dict[int, str]:
        """Load source text for specific pages only (fast path for training-only apply)."""
        targets = sorted({int(page) for page in page_nums})
        raw_pages = self._load_raw_pdf_pages(file_name, page_nums=targets)
        if raw_pages and any(text.strip() for text in raw_pages.values()):
            return raw_pages

        indexed = self._load_page_text_from_index(collection, file_name)
        return {page: indexed.get(page, "") for page in targets if page in indexed}

    def _load_page_text_for_file(self, collection, file_name: str) -> Dict[int, str]:
        """Load page source text for drawing-region chunking."""
        indexed = self._load_page_text_from_index(collection, file_name)
        if len(indexed) >= 5:
            return {
                page: DrawingRegionChunker._sanitize_page_source_text(text)
                for page, text in indexed.items()
                if text
            }

        raw_pages = self._load_raw_pdf_pages(file_name)
        if raw_pages and any(text.strip() for text in raw_pages.values()):
            return raw_pages
        try:
            pdf_path = storage._resolve_local_pdf_path(file_name)
        except Exception:
            return indexed
        if not pdf_path or not pdf_path.exists():
            return indexed

        from pdf_processor import PDFProcessor

        use_vision = len(indexed) < 10
        processor = PDFProcessor(use_vision=use_vision)
        pages_text, _ = processor.extract_text_from_pdf(pdf_path)
        merged = dict(pages_text)
        for page, text in indexed.items():
            sanitized = DrawingRegionChunker._sanitize_page_source_text(text)
            if len(sanitized) > len(merged.get(page, "")):
                merged[page] = sanitized
        return merged

    def apply_file(
        self,
        file_name: str,
        vector_store,
        use_cache: bool = True,
        force_refresh: bool = False,
        training_pages_only: bool = False,
    ) -> int:
        from enriched_metadata_loader import EnrichedMetadataLoader

        training_pages = {
            example["page"]
            for example in TRAINING_EXAMPLES
            if example["file_name"] == file_name
        }
        if training_pages_only and not training_pages:
            print(f"  No training pages for {file_name}, skipping")
            return 0

        vector_store.initialize_vectorstore()
        collection = vector_store.vectorstore._collection
        if training_pages_only:
            page_text = self._load_pages_for_chunking(
                collection, file_name, sorted(training_pages)
            )
        else:
            page_text = self._load_page_text_for_file(collection, file_name)
        enriched = EnrichedMetadataLoader.load_enriched_metadata(file_name) or {}
        existing_chunks = self._load_existing_chunks(collection, file_name)

        target_pages = sorted(page_text.keys())
        if training_pages_only:
            target_pages = [page for page in target_pages if page in training_pages]

        replacement_chunks: Dict[int, List[Dict]] = {}
        try:
            for page_num in tqdm(target_pages, desc=f"Drawing chunks {file_name}"):
                page_enriched = enriched.get(page_num)
                replacement_chunks[page_num] = self.chunker.build_page_chunks(
                    file_name=file_name,
                    page_num=page_num,
                    page_text=page_text[page_num],
                    page_enriched=page_enriched,
                    use_cache=use_cache,
                    force_refresh=force_refresh,
                )
        finally:
            self.chunker.close_pdf()

        if training_pages_only:
            kept = [chunk for chunk in existing_chunks if chunk["page"] not in replacement_chunks]
            new_chunks = kept[:]
            for page_num in sorted(replacement_chunks):
                new_chunks.extend(replacement_chunks[page_num])
        else:
            new_chunks = []
            for page_num in sorted(page_text.keys()):
                if page_num in replacement_chunks:
                    new_chunks.extend(replacement_chunks[page_num])
                else:
                    text = page_text[page_num].strip()
                    if text:
                        new_chunks.append(
                            {"text": text, "page": page_num, "chunk_id": 0}
                        )

        if not new_chunks:
            print(f"  No chunks produced for {file_name}")
            return 0

        existing = collection.get(where={"file_name": file_name}, include=["metadatas"])
        pdf_metadata = {}
        if existing["metadatas"]:
            sample = existing["metadatas"][0]
            pdf_metadata = {
                "project_name": sample.get("project_name", ""),
                "phase": sample.get("phase", ""),
                "engineer_of_record": sample.get("engineer_of_record", ""),
                "date": sample.get("date", ""),
                "total_pages": sample.get("total_pages", 0),
                "file_path": sample.get("file_path", ""),
            }

        # Re-number chunk ids per page for stable metadata.
        per_page_counter: Dict[int, int] = {}
        normalized_chunks: List[Dict] = []
        for chunk in new_chunks:
            page_num = int(chunk["page"])
            chunk_id = per_page_counter.get(page_num, 0)
            per_page_counter[page_num] = chunk_id + 1
            normalized_chunks.append({**chunk, "chunk_id": chunk_id})

        vector_store.delete_by_filename(file_name)
        pdf_data = {
            "file_name": file_name,
            "file_path": pdf_metadata.get("file_path", file_name),
            "metadata": pdf_metadata,
            "chunks": normalized_chunks,
            "total_pages": pdf_metadata.get("total_pages", max(page_text.keys()) if page_text else 0),
        }
        added = vector_store.add_documents([pdf_data])
        print(f"  Re-chunked {file_name}: {added} drawing-region chunks")
        return added

    def apply_all(
        self,
        vector_store,
        use_cache: bool = True,
        force_refresh: bool = False,
        training_pages_only: bool = False,
    ) -> int:
        total = 0
        for file_name in sorted(MetadataManager().get_active_pdf_file_names()):
            if training_pages_only and not any(
                example["file_name"] == file_name for example in TRAINING_EXAMPLES
            ):
                continue
            print(f"\nApplying drawing-region chunking: {file_name}")
            total += self.apply_file(
                file_name=file_name,
                vector_store=vector_store,
                use_cache=use_cache,
                force_refresh=force_refresh,
                training_pages_only=training_pages_only,
            )
        return total
