"""Build and persist LLM-synthesized project profiles for search and analysis."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

import config
from enhanced_topology_loader import EnhancedTopologyLoader
from metadata_manager import MetadataManager
from project_layout import collect_layout_signals


class ProjectProfileBuilder:
    """Generate project overview artifacts from GP sheets, topology, and layout signals."""

    PROFILE_VERSION = 1
    PROFILES_DIR = config.DATA_FOLDER / "project_profiles"
    MANIFEST_FILE = PROFILES_DIR / "manifest.json"

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.metadata_manager = MetadataManager()

    def list_target_files(self) -> List[str]:
        projects = self.metadata_manager.get_all_projects()
        if projects:
            return sorted(projects.keys())

        from enhanced_topology_loader import EnhancedTopologyLoader

        files = []
        for path in EnhancedTopologyLoader.list_deep_vision_topology_files():
            files.append(path.stem.replace("enhanced_topology_", "") + ".pdf")
        return sorted(files)

    def _gp_context_text(self, gp_items: List[Dict], max_chars: int = 6000) -> str:
        sections = []
        for item in gp_items:
            metadata = item.get("metadata", {}) or {}
            page = metadata.get("page")
            title = metadata.get("plan_sheet_title") or metadata.get("topology_sheet_title") or ""
            content = item.get("content", "") or ""
            for marker in (
                "[TOPOLOGY PROJECT]",
                "[TOPOLOGY SHEET]",
                "[TOPOLOGY DETAIL]",
                "[topology project]",
            ):
                if marker in content:
                    content = content.split(marker, 1)[0]
            content = re.sub(r"\s+", " ", content).strip()
            if not content:
                continue
            sections.append(f"--- General Plan Page {page} ({title}) ---\n{content[:2500]}")
        combined = "\n\n".join(sections)
        return combined[:max_chars]

    def _topology_context(self, file_name: str) -> str:
        topology = EnhancedTopologyLoader.load_topology(file_name)
        if not topology:
            return ""

        project = EnhancedTopologyLoader.infer_project_topology(topology)
        parts = [
            f"Project name: {topology.get('project_name', '')}",
            f"Structure type: {project.get('structure_type', '')}",
            f"Bridge type: {project.get('bridge_type', '')}",
            f"Superstructure: {project.get('superstructure_type', '')}",
            f"Substructure: {project.get('substructure_type', '')}",
            f"Foundation: {project.get('foundation_type', '')}",
            f"Project labels: {project.get('project_level_labels', '')}",
        ]
        summary = topology.get("summary") or {}
        if isinstance(summary, dict) and summary:
            parts.append(f"Topology summary: {json.dumps(summary)[:1200]}")
        return "\n".join(p for p in parts if p.split(": ", 1)[-1])

    def _generate_narrative(
        self,
        file_name: str,
        layout: Dict,
        topology_context: str,
        gp_context: str,
        project_meta: Optional[Dict],
    ) -> Dict:
        structured = {
            "span_count": layout.get("span_count"),
            "interior_bents": layout.get("interior_bents", []),
            "bridge_type": None,
            "structure_type": None,
            "superstructure": None,
            "foundation_system": None,
            "concrete_box_girder": bool(layout.get("concrete")),
            "is_bridge": bool(layout.get("bridge")),
        }

        for line in topology_context.splitlines():
            lower = line.lower()
            if lower.startswith("bridge type:"):
                structured["bridge_type"] = line.split(":", 1)[1].strip()
            elif lower.startswith("structure type:"):
                structured["structure_type"] = line.split(":", 1)[1].strip()
            elif lower.startswith("superstructure:"):
                structured["superstructure"] = line.split(":", 1)[1].strip()
            elif lower.startswith("foundation:"):
                structured["foundation_system"] = line.split(":", 1)[1].strip()

        meta_blob = ""
        if project_meta:
            meta_blob = (
                f"Indexed project: {project_meta.get('project_name', '')}\n"
                f"Phase: {project_meta.get('phase', '')}\n"
                f"Engineer: {project_meta.get('engineer_of_record', '')}\n"
                f"Total pages: {project_meta.get('total_pages', '')}\n"
            )

        layout_blob = (
            f"Inferred span count (from GP layout): {layout.get('span_count')}\n"
            f"Interior bents on GP: {layout.get('interior_bents', [])}\n"
            f"Concrete box girder: {layout.get('concrete')}\n"
            f"Bridge: {layout.get('bridge')}\n"
            f"GP pages: {[t.get('page') for t in layout.get('gp_titles', [])]}\n"
        )

        prompt = f"""You are a senior bridge engineer reviewing a plan set overview.

Write a project profile for search and design understanding. Use ONLY the evidence below.
If span count is provided from GP layout, treat it as authoritative unless evidence clearly contradicts it.
Do not invent sheet numbers or details not supported by the context.

Return JSON with exactly these keys:
- narrative: 2-3 paragraphs explaining bridge layout, structural system, and apparent design intent
- design_intent: 3-5 bullet strings (short) about why the bridge appears designed this way
- key_sheets: array of objects with page, title, role (e.g. "general plan layout")
- evidence: array of short strings citing what supports span count and bridge type
- uncertainties: array of short strings for anything unclear (empty if confident)

FILE: {file_name}

PROJECT METADATA:
{meta_blob}

LAYOUT SIGNALS:
{layout_blob}

TOPOLOGY SUMMARY:
{topology_context[:3000]}

GENERAL PLAN EXCERPTS:
{gp_context[:5000]}
"""

        profile_model = os.getenv("OPENAI_PROFILE_MODEL", "gpt-4o")

        try:
            response = self.client.chat.completions.create(
                model=profile_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You produce accurate structural engineering project profiles. "
                            "Respond with valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
        except Exception as exc:
            payload = {
                "narrative": (
                    f"Project profile for {file_name}. "
                    f"Span count {layout.get('span_count')}. "
                    f"Bridge type: {structured.get('bridge_type') or 'unknown'}."
                ),
                "design_intent": [],
                "key_sheets": layout.get("gp_titles", []),
                "evidence": [
                    f"GP layout interior bents: {layout.get('interior_bents', [])}",
                ],
                "uncertainties": [f"LLM narrative unavailable: {exc}"],
            }

        return {
            "structured": structured,
            "narrative": str(payload.get("narrative", "")).strip(),
            "design_intent": payload.get("design_intent") or [],
            "key_sheets": payload.get("key_sheets") or layout.get("gp_titles", []),
            "evidence": payload.get("evidence") or [],
            "uncertainties": payload.get("uncertainties") or [],
        }

    def build_profile(
        self,
        file_name: str,
        vector_store,
        keyword_candidates: Optional[List[Dict]] = None,
    ) -> Dict:
        layout = collect_layout_signals(file_name, vector_store, keyword_candidates)
        topology_context = self._topology_context(file_name)
        gp_context = self._gp_context_text(layout.get("gp_items", []))
        project_meta = self.metadata_manager.get_project_metadata(file_name)

        gp_titles: List[Dict] = []
        seen_pages = set()
        for title_row in layout.get("gp_titles", []):
            page = title_row.get("page")
            if page in seen_pages:
                continue
            seen_pages.add(page)
            gp_titles.append(title_row)

        generated = self._generate_narrative(
            file_name=file_name,
            layout={**layout, "gp_titles": gp_titles},
            topology_context=topology_context,
            gp_context=gp_context,
            project_meta=project_meta,
        )

        return {
            "version": self.PROFILE_VERSION,
            "built_at": datetime.utcnow().isoformat(),
            "file_name": file_name,
            "project_name": (project_meta or {}).get("project_name") or Path(file_name).stem,
            "source_pages": [t.get("page") for t in gp_titles if t.get("page")],
            "topology_version": EnhancedTopologyLoader.LAYERED_TOPOLOGY_VERSION,
            **generated,
        }

    def save_profile(self, profile: Dict) -> Path:
        self.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        file_name = profile.get("file_name", "unknown.pdf")
        stem = Path(file_name).stem
        output_path = self.PROFILES_DIR / f"profile_{stem}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2, ensure_ascii=False)
        return output_path

    def save_manifest(self, profiles: List[Dict]) -> Path:
        self.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": self.PROFILE_VERSION,
            "built_at": datetime.utcnow().isoformat(),
            "project_count": len(profiles),
            "projects": [
                {
                    "file_name": p.get("file_name"),
                    "project_name": p.get("project_name"),
                    "span_count": (p.get("structured") or {}).get("span_count"),
                    "bridge_type": (p.get("structured") or {}).get("bridge_type"),
                    "profile_path": str(
                        self.PROFILES_DIR / f"profile_{Path(p.get('file_name', '')).stem}.json"
                    ),
                }
                for p in profiles
            ],
        }
        with open(self.MANIFEST_FILE, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return self.MANIFEST_FILE

    @classmethod
    def load_all_profiles(cls) -> Dict[str, Dict]:
        """Load saved profiles keyed by file_name."""
        profiles: Dict[str, Dict] = {}
        if not cls.PROFILES_DIR.exists():
            return profiles

        for path in cls.PROFILES_DIR.glob("profile_*.json"):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    profile = json.load(handle)
                file_name = profile.get("file_name")
                if file_name:
                    profiles[file_name] = profile
            except Exception:
                continue
        return profiles

    @staticmethod
    def profile_to_search_text(profile: Dict) -> str:
        structured = profile.get("structured") or {}
        lines = [
            "[PROJECT PROFILE]",
            f"Project: {profile.get('project_name', '')} ({profile.get('file_name', '')})",
            f"Span Count: {structured.get('span_count', '')}",
            f"Structure Type: {structured.get('structure_type', '')}",
            f"Bridge Type: {structured.get('bridge_type', '')}",
            f"Superstructure: {structured.get('superstructure', '')}",
            f"Foundation System: {structured.get('foundation_system', '')}",
            f"Interior Bents: {', '.join(str(b) for b in structured.get('interior_bents', []))}",
            f"Concrete Box Girder: {structured.get('concrete_box_girder', '')}",
        ]

        narrative = profile.get("narrative", "")
        if narrative:
            lines.append("")
            lines.append("[PROJECT NARRATIVE]")
            lines.append(narrative)

        design_intent = profile.get("design_intent") or []
        if design_intent:
            lines.append("")
            lines.append("[DESIGN INTENT]")
            for item in design_intent:
                lines.append(f"- {item}")

        evidence = profile.get("evidence") or []
        if evidence:
            lines.append("")
            lines.append("[PROFILE EVIDENCE]")
            for item in evidence:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def run(self, vector_store, merge_to_vector_store: bool = True) -> Dict:
        vector_store.initialize_vectorstore()
        keyword_candidates = vector_store.get_keyword_search_candidates(
            limit=config.HYBRID_KEYWORD_CANDIDATE_LIMIT,
        )

        profiles: List[Dict] = []
        merged = 0

        for file_name in self.list_target_files():
            print(f"\nBuilding profile: {file_name}")
            try:
                profile = self.build_profile(file_name, vector_store, keyword_candidates)
                output_path = self.save_profile(profile)
                profiles.append(profile)
                span = (profile.get("structured") or {}).get("span_count")
                bridge = (profile.get("structured") or {}).get("bridge_type")
                print(f"  ✓ Saved {output_path.name} (spans={span}, type={bridge})")

                if merge_to_vector_store:
                    if vector_store.merge_project_profile(file_name, profile):
                        merged += 1
                        print("  ✓ Merged project overview into vector store")
            except Exception as exc:
                print(f"  Error: {exc}")

        manifest_path = self.save_manifest(profiles)
        return {
            "profile_count": len(profiles),
            "merged_count": merged,
            "manifest_path": str(manifest_path),
            "profiles_dir": str(self.PROFILES_DIR),
        }
