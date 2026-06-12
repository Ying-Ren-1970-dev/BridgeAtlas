"""Annotated drawing-region chunk training examples."""
from pathlib import Path

import config

TRAINING_DIR = config.DATA_FOLDER / "training"
CACHE_DIR = config.DATA_FOLDER / "drawing_chunks"

TRAINING_EXAMPLES = [
    {
        "id": "goldenwest_abutment_details_2",
        "file_name": "55-1119_GoldenwestOc_As_BuiltWM.pdf",
        "page": 17,
        "sheet_title": "ABUTMENT DETAILS No. 2",
        "sheet_group": "abutment",
        "expected_chunks": [
            {"label": "SECTION B-B", "view_type": "section"},
            {"label": "JOINT PROTECTION DETAIL", "view_type": "detail"},
            {"label": '24" ø STEEL PIPE PILE ELEVATION', "view_type": "elevation"},
            {"label": "SECTION Y-Y", "view_type": "section"},
            {"label": "AS BUILT SECTION Y-Y", "view_type": "section"},
        ],
    },
    {
        "id": "goldenwest_abutment_details_6",
        "file_name": "55-1119_GoldenwestOc_As_BuiltWM.pdf",
        "page": 21,
        "sheet_title": "ABUTMENT DETAILS No. 6",
        "sheet_group": "abutment",
        "expected_chunks": [
            {"label": "DETAIL 1", "view_type": "detail"},
            {"label": "DETAIL 2", "view_type": "detail"},
            {"label": "DETAIL 3", "view_type": "detail"},
            {"label": "SECTION A-A", "view_type": "section"},
            {"label": "SECTION B-B", "view_type": "section"},
            {"label": "SECTION C-C", "view_type": "section"},
            {"label": "DETAIL 5", "view_type": "detail"},
        ],
    },
    {
        "id": "goldenwest_chain_link_railing_details_1",
        "file_name": "55-1119_GoldenwestOc_As_BuiltWM.pdf",
        "page": 61,
        "sheet_title": "CHAIN LINK RAILING TYPE 2 (Mod) DETAILS No. 1",
        "sheet_group": "barrier",
        "expected_chunks": [
            {"label": "TYPICAL SECTION", "view_type": "section"},
            {"label": "ELEVATION", "view_type": "elevation"},
            {"label": "TYPICAL DETAILS", "view_type": "detail"},
            {"label": "TOP CABLE ANCHORAGE DETAIL", "view_type": "detail"},
            {"label": "POST ANCHORAGE DETAIL", "view_type": "detail"},
            {"label": "DETAIL 1", "view_type": "detail"},
        ],
    },
    {
        "id": "westminster_abutment_3_layout",
        "file_name": "55-1127_WestminsterOC_As_Built.pdf",
        "page": 11,
        "sheet_title": "ABUTMENT 3 LAYOUT",
        "sheet_group": "abutment",
        "expected_chunks": [
            {"label": "AS BUILT", "view_type": "as_built"},
            {"label": "PLAN", "view_type": "plan"},
            {"label": "ELEVATION", "view_type": "elevation"},
        ],
    },
    {
        "id": "magnolia_bent_layout_1",
        "file_name": "55-1121_MagnoliaOc_As_BuiltWM.pdf",
        "page": 10,
        "sheet_title": "BENT LAYOUT No. 1",
        "sheet_group": "bent",
        "expected_chunks": [
            {"label": "PLAN STAGE 1", "view_type": "plan"},
            {"label": "ELEVATION STAGE 1", "view_type": "elevation"},
            {"label": "SECTION B-B", "view_type": "section"},
            {"label": "SECTION A-A", "view_type": "section"},
        ],
    },
    {
        "id": "fresno_bent_details_2",
        "file_name": "Fresno 180 Ramp.pdf",
        "page": 12,
        "sheet_title": "BENT DETAILS NO. 2",
        "sheet_group": "bent",
        "expected_chunks": [
            {"label": "SECTION A-A", "view_type": "section"},
            {"label": "SECTION B-B", "view_type": "section"},
            {"label": "BENT CAP CAMBER DIAGRAM", "view_type": "diagram"},
            {"label": "PLAN", "view_type": "plan"},
            {"label": "ELEVATION", "view_type": "elevation"},
        ],
    },
]
