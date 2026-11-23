"""
Configuration module for the TCS Autonomous Asset Inspection Vision System.
Powered by YOLOv8 real-time object detection and asset-triggered inspection.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os

# ── Paths & Directories ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
ANNOTATED_DIR = os.path.join(BASE_DIR, "annotated")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
TCS_LOGO_PATH = os.path.join(BASE_DIR, "tcs_logo.png")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")

BATCH_SIZE = 4  # Generate inspection report after every 4 assets

# ── YOLO Thresholds ──────────────────────────────────────────────────────
# Base confidence threshold for YOLO detections (0.0‑1.0). Lower to be more permissive.
YOLO_CONF_THRESH = 0.15
# IoU threshold for non‑max suppression (0.0‑1.0). Higher = stricter duplicate removal.
YOLO_IOU_THRESH = 0.35

# ── Detection Confidence Settings ───────────────────────────────────────
TAG_DETECT_CONF = 0.05  # Confidence for asset tag detection
ANOMALY_DETECT_CONF = 0.15  # Confidence for defect detection
ROI_DETECT_CONF = 0.20  # Confidence for ROI inspection
NMS_IOU_THRESH = 0.35  # IoU threshold for NMS deduplication

# ── Asset Classes & Parameters ────────────────────────────────────────────────
ASSET_TAG_MAP = {
    "tag_TR1": ("TR-1", "Transformer"),
    "tag_TR2": ("TR-2", "Transformer"),
    "tag_CB1": ("CB-1", "Circuit Breaker"),
    "tag_CS02": ("CS-02", "Cable Section"),
}

ASSET_TYPE_MAP = {
    "TR-1": "Transformer",
    "TR-2": "Transformer",
    "CB-1": "Circuit Breaker",
    "CS-02": "Cable Section",
}

ASSET_ID_PATTERNS = {
    "TR-1": r"\bTR[- ]?0?1\b",
    "TR-2": r"\bTR[- ]?0?2\b",
    "CB-1": r"\bCB[- ]?0?1\b",
    "CS-02": r"\bCS[- ]?0?2\b",
}

# Inspection Parameters per Asset Type
INSPECTION_PARAMS = {
    "Transformer": [
        "Oil Leakage",
        "Cracks",
        "Oil Gauge Level",
        "Bushing Condition",
        "Surface Integrity",
    ],
    "Circuit Breaker": [
        "Physical Damage",
        "Loose Connections",
        "Housing Integrity",
        "Terminal Condition",
        "Mechanism Integrity",
    ],
    "Cable Section": [
        "Insulation Damage & Cuts",
        "Burn Marks & Scorch",
        "Loose / Disconnected Cable",
        "Exposed Conductors",
        "Cable Support & Clamping",
    ],
}

# Parameter mapping for YOLO defect classes
DEFECT_PARAMETER_MAP = {
    "oil_leakage": {
        "param": "Oil Leakage",
        "severity": "Critical",
        "name": "Oil Leakage Detected",
        "description": "Visible oil leakage on/around transformer tank and puddling at base frame.",
    },
    "crack": {
        "param": "Cracks",
        "severity": "Major",
        "name": "Structural Crack",
        "description": "Visible crack / fissure along the tank weld joint.",
    },
    "insulation_damage": {
        "param": "Insulation Damage & Cuts",
        "severity": "Critical",
        "name": "Insulation Damage",
        "description": "Severe insulation tear and shredded sheathing exposing cable core.",
    },
    "cable_damage": {
        "param": "Insulation Damage & Cuts",
        "severity": "Critical",
        "name": "Insulation Damage / Cut",
        "description": "Deep cuts and insulation tears exposing internal cable structure.",
    },
    "burn_mark": {
        "param": "Burn Marks & Scorch",
        "severity": "Critical",
        "name": "Burn / Cable Damage",
        "description": "Severe heat scorch, charring marks, and surface cable damage.",
    },
    "loose_connection": {
        "param": "Loose / Disconnected Cable",
        "severity": "Major",
        "name": "Loose Connection",
        "description": "Disconnected or loose cable terminal hanging from support tray.",
    },
    "physical_damage": {
        "param": "Physical Damage",
        "severity": "Major",
        "name": "Physical Damage",
        "description": "Mechanical deformation or damage to housing structure.",
    },
}

# Known baseline oil gauge readings for transformers
GAUGE_READINGS = {
    "1.1 Transformer-1 (Healthy).png": 90,
    "1.2 Transformer-1 (Faulty).png": 35,
    "1.3 Transformer-2 (Healthy).png": 94,
}


# ── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class DefectFinding:
    """Represents a single defect detected during inspection."""
    defect_type: str
    severity: str       # "Critical", "Major", "Minor"
    location: str
    description: str
    confidence: float = 0.0
    bbox: Optional[List[int]] = None


@dataclass
class InspectionResult:
    """Represents the complete inspection result for one asset."""
    asset_id: str
    asset_type: str
    timestamp: str
    image_path: str
    annotated_image_path: str = ""
    overall_condition: str = "Healthy"   # "Healthy" or "Faulty"
    oil_gauge_reading: Optional[int] = None
    findings: list = field(default_factory=list)
    detected_components: list = field(default_factory=list)
    inference_time_ms: float = 0.0
