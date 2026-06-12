"""Auto-training for drawing-region chunking from annotated example sheets."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import config
from chunk_inspector import inspect_page_chunks
from drawing_chunk_examples import TRAINING_EXAMPLES
from drawing_region_chunker import DrawingRegionChunker, DrawingRegionSpec

from drawing_chunk_examples import CACHE_DIR, TRAINING_DIR

MODEL_PATH = TRAINING_DIR / "drawing_chunk_model.json"


@dataclass
class ExampleTrainingResult:
    example_id: str
    file_name: str
    page: int
    expected_count: int
    detected_count: int
    matched_labels: List[str]
    missing_labels: List[str]
    extra_labels: List[str]
    passed: bool
    detected_chunks: List[Dict] = field(default_factory=list)


@dataclass
class DrawingChunkModel:
    version: int
    trained_at: str
    examples: List[Dict]
    training_results: List[Dict]
    passed_examples: int
    total_examples: int
    ready_for_apply: bool
    prompt_version: str = "drawing-region-v1"

    def to_dict(self) -> Dict:
        return asdict(self)


class DrawingChunkTrainer:
    """Train and validate drawing-region chunk detection on 6 annotated sheets."""

    def __init__(self, chunker: Optional[DrawingRegionChunker] = None):
        self.chunker = chunker or DrawingRegionChunker()
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_label(label: str) -> str:
        return " ".join((label or "").upper().split())

    def _label_is_covered(
        self,
        expected_label: str,
        expected_view: str,
        detected: Sequence[DrawingRegionSpec],
    ) -> bool:
        expected_norm = self._normalize_label(expected_label)
        for item in detected:
            detected_norm = self._normalize_label(item.label)
            if (
                expected_norm in detected_norm
                or detected_norm in expected_norm
                or expected_norm == detected_norm
            ):
                return True
            if expected_norm == "ELEVATION" and item.view_type == "elevation":
                return True
            if expected_view and item.view_type == expected_view:
                if expected_view in {"section", "detail", "plan", "diagram", "as_built"}:
                    if expected_norm.split()[0] in detected_norm:
                        return True
        return False

    def _compare_labels(
        self,
        expected: Sequence[Dict],
        detected: Sequence[DrawingRegionSpec],
    ) -> tuple[List[str], List[str], List[str]]:
        expected_labels = [self._normalize_label(item["label"]) for item in expected]
        detected_labels = [self._normalize_label(item.label) for item in detected]

        matched: List[str] = []
        missing: List[str] = []
        for item in expected:
            label = self._normalize_label(item["label"])
            if self._label_is_covered(label, str(item.get("view_type") or ""), detected):
                matched.append(label)
            else:
                missing.append(label)

        extra = [
            label for label in detected_labels
            if not any(
                label in expected_label or expected_label in label
                for expected_label in expected_labels
            )
        ]
        return matched, missing, extra

    def train(self, use_cache: bool = True) -> DrawingChunkModel:
        """Run vision detection on all 6 training pages and validate labels."""
        print("=" * 72)
        print("DRAWING CHUNK AUTO-TRAINING (6 example sheets)")
        print("=" * 72)

        results: List[ExampleTrainingResult] = []
        for example in TRAINING_EXAMPLES:
            print(f"\nTraining example: {example['id']} ({example['file_name']} p{example['page']})")
            detected = self.chunker.detect_page_regions(
                file_name=example["file_name"],
                page_num=example["page"],
                use_cache=use_cache,
                force_refresh=True,
            )
            matched, missing, extra = self._compare_labels(
                example["expected_chunks"], detected
            )
            passed = not missing
            result = ExampleTrainingResult(
                example_id=example["id"],
                file_name=example["file_name"],
                page=example["page"],
                expected_count=len(example["expected_chunks"]),
                detected_count=len(detected),
                matched_labels=matched,
                missing_labels=missing,
                extra_labels=extra,
                passed=passed,
                detected_chunks=[spec.to_dict() for spec in detected],
            )
            results.append(result)
            status = "PASS" if passed else "FAIL"
            print(
                f"  {status}: detected {result.detected_count}/{result.expected_count} "
                f"(missing={missing or 'none'}, extra={extra or 'none'})"
            )

        passed_count = sum(1 for result in results if result.passed)
        model = DrawingChunkModel(
            version=1,
            trained_at=datetime.now(timezone.utc).isoformat(),
            examples=TRAINING_EXAMPLES,
            training_results=[asdict(result) for result in results],
            passed_examples=passed_count,
            total_examples=len(TRAINING_EXAMPLES),
            ready_for_apply=passed_count >= 4,
        )
        MODEL_PATH.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
        print("\n" + "=" * 72)
        print(
            f"TRAINING COMPLETE: {passed_count}/{len(TRAINING_EXAMPLES)} examples passed"
        )
        print(f"Model saved: {MODEL_PATH}")
        print(f"Ready for apply: {model.ready_for_apply}")
        print("=" * 72)
        return model

    @staticmethod
    def load_model() -> Optional[DrawingChunkModel]:
        if not MODEL_PATH.exists():
            return None
        payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        return DrawingChunkModel(**payload)

    def baseline_index_report(self, collection=None) -> None:
        """Show current index chunk quality on training pages before/after apply."""
        print("\nBaseline index inspection (current chunks):")
        for example in TRAINING_EXAMPLES:
            labels = [chunk["label"] for chunk in example["expected_chunks"]]
            report = inspect_page_chunks(
                file_name=example["file_name"],
                page=example["page"],
                expected_labels=labels,
                collection=collection,
            )
            status = "PASS" if report.passed else "FAIL"
            print(
                f"  {example['id']}: {status} "
                f"(chunks={report.chunk_count}, expected={len(labels)}, mixed={report.mixed_chunks})"
            )
