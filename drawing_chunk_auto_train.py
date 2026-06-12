"""Auto-train drawing-region rechunking and validate indexed chunks on training pages."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from chunk_inspector import format_report, inspect_page_chunks
from drawing_chunk_examples import TRAINING_EXAMPLES
from drawing_chunk_training import (
    CACHE_DIR,
    TRAINING_DIR,
    DrawingChunkModel,
    DrawingChunkTrainer,
)
from drawing_region_chunker import DrawingChunkApplicator

VALIDATION_REPORT_PATH = TRAINING_DIR / "rechunk_validation_report.json"


@dataclass
class PageValidationResult:
    example_id: str
    file_name: str
    page: int
    passed: bool
    chunk_count: int
    expected_count: int
    mixed_chunks: int
    missing_labels: List[str] = field(default_factory=list)


@dataclass
class FileValidationResult:
    file_name: str
    has_training_pages: bool
    pages_passed: int
    pages_total: int
    passed: bool
    pages: List[PageValidationResult] = field(default_factory=list)


@dataclass
class AutoTrainRunReport:
    trained_at: str
    vision_passed: int
    vision_total: int
    vision_ready: bool
    files: List[FileValidationResult] = field(default_factory=list)

    @property
    def all_training_pages_passed(self) -> bool:
        checked = [item for item in self.files if item.has_training_pages]
        return bool(checked) and all(item.passed for item in checked)

    def to_dict(self) -> Dict:
        return {
            "trained_at": self.trained_at,
            "vision_passed": self.vision_passed,
            "vision_total": self.vision_total,
            "vision_ready": self.vision_ready,
            "all_training_pages_passed": self.all_training_pages_passed,
            "files": [asdict(item) for item in self.files],
        }


def training_examples_for_file(file_name: str) -> List[Dict]:
    """Return annotated training examples for one indexed PDF."""
    return [example for example in TRAINING_EXAMPLES if example["file_name"] == file_name]


def has_training_pages(file_name: str) -> bool:
    return bool(training_examples_for_file(file_name))


class DrawingChunkAutoTrainer:
    """
    Orchestrate vision training, rechunk apply, and post-apply validation.

    Vision training uses engineer-annotated pages in drawing_chunk_examples.py.
    Validation uses the same expectations via inspect_page_chunks.
    """

    def __init__(self, vector_store=None):
        self.trainer = DrawingChunkTrainer()
        self.applicator = DrawingChunkApplicator()
        self.vector_store = vector_store
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def run_vision_training(
        self,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> DrawingChunkModel:
        """Train vision region detection on all annotated example sheets."""
        if force_refresh:
            use_cache = False
        return self.trainer.train(use_cache=use_cache)

    def validate_file(
        self,
        file_name: str,
        collection=None,
        verbose: bool = False,
    ) -> FileValidationResult:
        """Inspect indexed chunks on every training page for one PDF."""
        examples = training_examples_for_file(file_name)
        if not examples:
            return FileValidationResult(
                file_name=file_name,
                has_training_pages=False,
                pages_passed=0,
                pages_total=0,
                passed=True,
            )

        page_results: List[PageValidationResult] = []
        for example in examples:
            labels = [chunk["label"] for chunk in example["expected_chunks"]]
            report = inspect_page_chunks(
                file_name=file_name,
                page=example["page"],
                expected_labels=labels,
                collection=collection,
            )
            if verbose:
                print(format_report(report))
            page_results.append(
                PageValidationResult(
                    example_id=example["id"],
                    file_name=report.file_name,
                    page=report.page,
                    passed=report.passed,
                    chunk_count=report.chunk_count,
                    expected_count=len(labels),
                    mixed_chunks=report.mixed_chunks,
                    missing_labels=list(report.missing_labels),
                )
            )

        pages_passed = sum(1 for item in page_results if item.passed)
        return FileValidationResult(
            file_name=file_name,
            has_training_pages=True,
            pages_passed=pages_passed,
            pages_total=len(page_results),
            passed=pages_passed == len(page_results),
            pages=page_results,
        )

    def apply_with_validation(
        self,
        file_name: str,
        vector_store,
        *,
        max_retries: int = 1,
        force_refresh: bool = False,
        validate: bool = True,
        verbose: bool = False,
    ) -> tuple[int, FileValidationResult]:
        """
        Apply drawing-region chunking, optionally retry with fresh vision, then validate.
        """
        if not vector_store.vectorstore:
            vector_store.initialize_vectorstore()
        collection = vector_store.vectorstore._collection

        added = 0
        validation = FileValidationResult(
            file_name=file_name,
            has_training_pages=has_training_pages(file_name),
            pages_passed=0,
            pages_total=0,
            passed=True,
        )

        attempts = max(1, max_retries + 1)
        refresh = force_refresh
        for attempt in range(attempts):
            if attempt:
                print(f"  Rechunk retry {attempt}/{max_retries} with fresh vision cache")
                refresh = True

            added = self.applicator.apply_file(
                file_name=file_name,
                vector_store=vector_store,
                use_cache=not refresh,
                force_refresh=refresh,
                training_pages_only=False,
            )
            print(f"  Rechunk applied: {added} region chunks")

            if not validate:
                validation = FileValidationResult(
                    file_name=file_name,
                    has_training_pages=has_training_pages(file_name),
                    pages_passed=0,
                    pages_total=0,
                    passed=True,
                )
                break

            validation = self.validate_file(
                file_name,
                collection=collection,
                verbose=verbose,
            )
            self._print_file_validation(validation)
            if validation.passed or attempt == attempts - 1:
                break

        return added, validation

    def run_pipeline_training(
        self,
        file_names: Sequence[str],
        vector_store,
        *,
        auto_train_vision: bool = True,
        validate: bool = True,
        rechunk_retries: int = 1,
        force_refresh_vision: bool = False,
        verbose: bool = False,
    ) -> AutoTrainRunReport:
        """
        Train vision once, rechunk each file with validation/retries, save report.
        """
        model = None
        if auto_train_vision:
            model = self.run_vision_training(
                use_cache=not force_refresh_vision,
                force_refresh=force_refresh_vision,
            )
        else:
            model = DrawingChunkTrainer.load_model()
            if not model:
                raise RuntimeError(
                    "No drawing chunk model found. Run with auto_train_vision=True "
                    "or: python main.py train-drawing-chunks"
                )

        file_results: List[FileValidationResult] = []
        for file_name in file_names:
            print(f"\n--- Auto rechunk: {file_name} ---")
            _, validation = self.apply_with_validation(
                file_name=file_name,
                vector_store=vector_store,
                max_retries=rechunk_retries,
                force_refresh=force_refresh_vision,
                validate=validate,
                verbose=verbose,
            )
            file_results.append(validation)

        report = AutoTrainRunReport(
            trained_at=datetime.now(timezone.utc).isoformat(),
            vision_passed=model.passed_examples,
            vision_total=model.total_examples,
            vision_ready=model.ready_for_apply,
            files=file_results,
        )
        VALIDATION_REPORT_PATH.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )
        self._print_summary(report)
        return report

    @staticmethod
    def _print_file_validation(result: FileValidationResult) -> None:
        if not result.has_training_pages:
            print("  Validation: skipped (no training pages for this PDF)")
            return
        status = "PASS" if result.passed else "FAIL"
        print(
            f"  Validation: {status} "
            f"({result.pages_passed}/{result.pages_total} training pages)"
        )
        for page in result.pages:
            mark = "PASS" if page.passed else "FAIL"
            missing = ", ".join(page.missing_labels) if page.missing_labels else "none"
            print(
                f"    {mark} {page.example_id} p{page.page}: "
                f"chunks={page.chunk_count}/{page.expected_count}, "
                f"mixed={page.mixed_chunks}, missing={missing}"
            )

    @staticmethod
    def _print_summary(report: AutoTrainRunReport) -> None:
        print("\n" + "=" * 72)
        print("RECHUNK AUTO-TRAIN SUMMARY")
        print("=" * 72)
        print(
            f"Vision training: {report.vision_passed}/{report.vision_total} examples passed "
            f"(ready_for_apply={report.vision_ready})"
        )
        checked = [item for item in report.files if item.has_training_pages]
        if checked:
            passed_files = sum(1 for item in checked if item.passed)
            print(
                f"Indexed validation: {passed_files}/{len(checked)} PDFs passed all training pages"
            )
        else:
            print("Indexed validation: no training pages in scope")
        print(f"Report saved: {VALIDATION_REPORT_PATH}")
        print("=" * 72)
