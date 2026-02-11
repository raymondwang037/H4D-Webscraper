#!/usr/bin/env python3
"""
USPTO ODP Patent File Wrapper incremental scraper (RapidRotor-AI landscape)

What it does:
- Choose a keyword "area" via CLI (rotor-wake, meshing, AI-assisted CFD, etc.)
- Pull the NEXT N records from ODP results (offset-based via iterator slicing)
- Append to a master CSV (dedupe by ip_number)
- Rescore the ENTIRE master dataset after each append
- Write:
  - master CSV (overwrite if not locked)
  - timestamped snapshot (always)
  - timestamped top-N shortlist (always)

Requirements:
  pip install pyuspto

Env:
  PowerShell:
    $env:USPTO_ODP_API_KEY="YOUR_KEY"

Examples:
  # Start a master file (first 200)
  python run.py --area rotor_wake --years 10 --add 200 --master outputs\\odp_master.csv --top 500

  # Add next 200 from same area (auto offset = current master size)
  python run.py --area rotor_wake --years 10 --add 200 --master outputs\\odp_master.csv --top 500

  # Pull a broader set for an area (don’t require CFD terms in the query)
  python run.py --area ai_cfd --broad --years 10 --add 200 --master outputs\\odp_master.csv

  # Switch areas and keep appending into the SAME master
  python run.py --area meshing --years 10 --add 200 --master outputs\\odp_master.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pyUSPTO.clients.patent_data import PatentDataClient


# ----------------------------
# Keyword libraries (areas)
# ----------------------------

AREA_TERMS: Dict[str, List[str]] = {
    # Rotorcraft-specific aerodynamics and wake physics
    "rotor_wake": [
        "rotorcraft", "helicopter", "tiltrotor", "VTOL", "eVTOL",
        "rotor", "rotor blade", "blade",
        "rotor wake", "wake interaction", "induced velocity", "inflow model", "dynamic inflow",
        "blade vortex interaction", "BVI", "tip vortex", "vortex ring state",
        "hover", "forward flight", "transition flight", "maneuvering flight",
        "rotor aeroacoustics", "rotor noise", "tonal noise", "broadband noise",
        "ground effect", "ship airwake", "brownout", "dust cloud", "degraded visual environment", "DVE",
    ],

    # Unsteady aerodynamics and turbulence modeling
    "unsteady_turbulence": [
        "unsteady aerodynamics", "unsteady flow", "time accurate simulation", "time-accurate simulation",
        "URANS", "DES", "DDES", "IDDES",
        "turbulence model", "k-omega", "k-epsilon", "SST", "Spalart Allmaras", "Spalart–Allmaras",
        "transition model", "laminar turbulent transition", "laminar-turbulent transition",
    ],

    # Mesh and geometry automation
    "meshing": [
        "mesh generation", "automatic meshing", "unstructured mesh",
        "overset mesh", "chimera grid", "sliding mesh", "moving mesh",
        "adaptive mesh refinement", "AMR", "h-adaptation", "p-adaptation",
        "mesh morphing", "shape morphing", "geometry parameterization",
        "watertight geometry", "CAD repair", "surface triangulation",
    ],

    # Rotor motion, rotating frames, and blade kinematics (non-propulsive)
    "rotor_motion": [
        "rotating frame", "MRF", "multiple reference frame",
        "rigid body motion", "six degree of freedom", "6-DOF",
        "blade pitch", "collective", "cyclic", "flapping", "lead lag", "lead-lag",
        "aeroelastic", "fluid structure interaction", "FSI", "coupled simulation",
        "flutter", "vibration", "loads prediction", "hub loads",
    ],

    # Design optimization and speeding up the loop
    "optimization": [
        "adjoint method", "adjoint solver",
        "design optimization", "shape optimization", "multi objective optimization", "multi-objective optimization",
        "surrogate model", "response surface", "reduced order model", "ROM",
        "model order reduction", "POD", "Galerkin",
        "uncertainty quantification", "UQ", "robust design",
    ],

    # HPC / acceleration language without saying CFD
    "hpc_accel": [
        "reduced computation time", "real time simulation", "near real time",
        "parallel computing", "MPI", "GPU acceleration", "heterogeneous computing",
        "solver acceleration", "preconditioner", "multigrid", "AMG",
        "convergence acceleration", "residual minimization",
    ],

    # Data-driven / AI-assisted CFD
    "ai_cfd": [
        "physics informed neural network", "PINN",
        "neural operator", "Fourier neural operator", "FNO",
        "surrogate CFD", "emulator", "learned turbulence model",
        "data driven model", "machine learning",  # pair with aero/rotor terms via aero anchor
        "digital twin",
        "reduced fidelity correction", "multi fidelity fusion", "multi-fidelity fusion",
    ],

    # Fixed-wing aerodynamic design terms that often co-occur with CFD
    "fixed_wing": [
        "aircraft", "airplane", "wing", "airfoil", "airframe",
        "high lift", "slat", "flap", "separation control",
        "boundary layer control", "flow control", "vortex generator",
        "drag reduction", "laminar flow", "natural laminar flow", "NLF",
        "aeroacoustic", "airframe noise", "landing gear",
    ],
}

# Generic anchors (used to keep search “on mission”)
CFD_ANCHOR = [
    "computational fluid dynamics", "CFD", "Navier Stokes", "flow solver", "finite volume",
    "mesh", "grid", "turbulence", "aerodynamics",
]

AERO_ANCHOR = [
    "aircraft", "airplane", "airframe", "wing", "airfoil",
    "rotorcraft", "helicopter", "rotor", "blade", "VTOL", "UAV",
]

PROPULSION_EXCLUDE = [
    "propulsion", "engine", "turbine", "combustion", "combustor",
    "nozzle", "inlet", "exhaust", "afterburner", "compressor", "fuel",
]


# ----------------------------
# Query builder
# ----------------------------

def build_query(area: str, *, broad: bool) -> str:
    """
    area: one of AREA_TERMS keys
    broad:
      - False: require CFD anchors AND aero anchors AND area terms (best precision)
      - True:  require aero anchors AND area terms (wider net; scoring will help)
    """
    area_terms = AREA_TERMS[area]

    area_block = f"({ ' OR '.join(area_terms) })"
    aero_block = f"({ ' OR '.join(AERO_ANCHOR) })"
    cfd_block = f"({ ' OR '.join(CFD_ANCHOR) })"
    not_block = f"NOT ({ ' OR '.join(PROPULSION_EXCLUDE) })"

    if broad:
        return f"{aero_block} AND {area_block} AND {not_block}"
    return f"{cfd_block} AND {aero_block} AND {area_block} AND {not_block}"


# ----------------------------
# Scoring (global, not area-specific)
# ----------------------------

def score_text(text: str) -> Tuple[int, str]:
    t = (text or "").lower()
    score = 0
    why: List[str] = []

    def hit(pat: str, pts: int, label: str):
        nonlocal score
        if re.search(pat, t):
            score += pts
            why.append(label)

    # CFD signals
    hit(r"computational fluid dynamics", 8, "CFD phrase")
    hit(r"\bcfd\b", 6, "CFD acronym")
    hit(r"navier[-\s]?stokes", 5, "Navier–Stokes")
    hit(r"\burans\b|\bdes\b|\bddes\b|\biddes\b", 3, "unsteady turbulence acronyms")
    hit(r"\brans\b|reynolds[-\s]?averaged", 3, "RANS")
    hit(r"\bles\b|large eddy simulation", 3, "LES")
    hit(r"turbulence model|k[-\s]?omega|k[-\s]?epsilon|sst|spalart", 2, "turbulence model names")
    hit(r"mesh generation|unstructured mesh|overset mesh|chimera|adaptive mesh|amr|grid refinement", 2, "meshing")
    hit(r"solver|convergence|residual|multigrid|amg|precondition", 2, "solver acceleration")

    # Aero/rotorcraft context
    hit(r"rotorcraft|helicopter|tiltrotor|evtol|vtol", 3, "rotorcraft")
    hit(r"rotor wake|wake interaction|induced velocity|dynamic inflow|inflow model", 3, "wake/inflow")
    hit(r"blade[-\s]?vortex interaction|\bbvi\b|tip vortex|vortex ring state", 3, "BVI/vortices")
    hit(r"hover|forward flight|transition flight|maneuver", 2, "flight regime")
    hit(r"aeroacoustic|rotor noise|tonal noise|broadband noise|airframe noise", 2, "noise")
    hit(r"boundary layer|separation|stall|laminar|transition|drag reduction|vortex generator", 2, "aero physics")

    # Optimization + AI assist
    hit(r"adjoint", 3, "adjoint")
    hit(r"shape optimization|design optimization|multi[-\s]?objective|surrogate model|response surface|rom|reduced order", 2, "optimization/ROM")
    hit(r"uncertainty quantification|\buq\b|robust design", 2, "UQ")
    hit(r"physics[-\s]?informed neural network|\bpinn\b|neural operator|fourier neural operator|\bfno\b", 3, "AI/PINN/FNO")
    hit(r"surrogate cfd|emulator|learned turbulence|multi[-\s]?fidelity|digital twin", 2, "AI surrogate/multifidelity")

    # Mild penalty if propulsion slips in (query has NOT block, but keep safety)
    if re.search(r"propulsion|engine|turbine|combust|nozzle|compressor|fuel", t):
        score -= 10
        why.append("propulsion penalty")

    return score, ", ".join(why)


# ----------------------------
# Row model + IO
# ----------------------------

@dataclass
class Row:
    ip_number: str
    title: str
    filing_date: str
    inventors: str
    assignees: str
    contact_details: str
    abstract: str
    score: int
    why_scored: str
    area: str


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def safe_join(parts: List[str], sep: str = "; ") -> str:
    return sep.join([p.strip() for p in parts if p and p.strip()])


def extract_fields(app: Any) -> Tuple[str, str, str, str, str, str, str]:
    """
    Based on your confirmed model output:
      app.application_number_text
      app.application_meta_data.filing_date
      app.application_meta_data.first_inventor_name
      app.application_meta_data.first_applicant_name

    Title/abstract may be absent depending on what ODP returns for this endpoint/result.
    We'll probe reasonable names but won't block output if missing.
    """
    ip_number = safe_str(getattr(app, "application_number_text", "")).strip()
    amd = getattr(app, "application_meta_data", None)

    filing_date = ""
    inventors = ""
    assignees = ""
    contact_details = ""
    title = ""
    abstract = ""

    if amd is not None:
        filing_date = safe_str(getattr(amd, "filing_date", "")).strip()
        inventors = safe_str(getattr(amd, "first_inventor_name", "")).strip()
        assignees = safe_str(getattr(amd, "first_applicant_name", "")).strip()

        # Best-effort probes (may or may not exist in your pyUSPTO version)
        for attr in ("invention_title", "inventionTitle", "title", "invention_title_text", "title_text"):
            v = getattr(amd, attr, None)
            if v:
                title = safe_str(v).strip()
                break

        for attr in ("abstract_text", "abstractText", "abstract"):
            v = getattr(amd, attr, None)
            if v:
                abstract = safe_str(v).strip()
                break

        contact_details = safe_join([
            safe_str(getattr(amd, "docket_number", "")).strip(),
            safe_str(getattr(amd, "customer_number", "")).strip(),
        ])

    return ip_number, title, filing_date, abstract, inventors, assignees, contact_details


def read_master(path: Path) -> Dict[str, Row]:
    rows: Dict[str, Row] = {}
    if not path.exists():
        return rows

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            ip = (d.get("ip_number") or "").strip()
            if not ip:
                continue
            rows[ip] = Row(
                ip_number=ip,
                title=(d.get("title") or "").strip(),
                filing_date=(d.get("filing_date") or "").strip(),
                inventors=(d.get("inventors") or "").strip(),
                assignees=(d.get("assignees") or "").strip(),
                contact_details=(d.get("contact_details") or "").strip(),
                abstract=(d.get("abstract") or "").strip(),
                score=int((d.get("score") or "0") or 0),
                why_scored=(d.get("why_scored") or "").strip(),
                area=(d.get("area") or "").strip(),
            )
    return rows


def write_csv(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ip_number", "title", "filing_date", "inventors", "assignees",
            "contact_details", "abstract", "score", "why_scored", "area"
        ])
        for r in rows:
            w.writerow([
                r.ip_number, r.title, r.filing_date, r.inventors, r.assignees,
                r.contact_details, r.abstract, r.score, r.why_scored, r.area
            ])


# ----------------------------
# ODP pull (reliable next-N via iterator slicing)
# ----------------------------

def fetch_next_batch(
    client: PatentDataClient,
    *,
    q: str,
    from_date: str,
    to_date: str,
    start_offset: int,
    add: int,
    page_size: int,
) -> List[Any]:
    it = client.paginate_applications(
        query=q,
        filing_date_from_q=from_date,
        filing_date_to_q=to_date,
        limit=page_size,  # <-- was 100
    )
    return list(islice(it, start_offset, start_offset + add))



# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-size", type=int, default=25, help="ODP page size (lower avoids 6MB 413)")
    ap.add_argument(
        "--area",
        choices=list(AREA_TERMS.keys()),
        default="rotor_wake",
        help="Keyword area to drive the ODP search query",
    )
    ap.add_argument(
        "--broad",
        action="store_true",
        help="Do not require CFD anchors in the query (wider net; scoring ranks relevance)",
    )
    ap.add_argument("--years", type=int, default=10, help="Lookback window on filing date")
    ap.add_argument("--add", type=int, default=200, help="How many NEW records to pull and append")
    ap.add_argument("--offset", type=int, default=None, help="Start offset (default: current master row count)")
    ap.add_argument("--master", type=str, default="outputs/odp_master.csv", help="Master CSV path")
    ap.add_argument("--top", type=int, default=500, help="Write a top-N shortlist snapshot")
    args = ap.parse_args()

    api_key = os.getenv("USPTO_ODP_API_KEY") or os.getenv("USPTO_API_KEY") or os.getenv("X_API_KEY")
    if not api_key:
        raise RuntimeError("Missing USPTO ODP key. Set USPTO_ODP_API_KEY in your environment.")

    master_path = Path(args.master)
    existing = read_master(master_path)

    start_offset = args.offset if args.offset is not None else len(existing)

    today = dt.date.today()
    from_dt = today.replace(year=today.year - args.years)
    from_date = from_dt.isoformat()
    to_date = today.isoformat()

    q = build_query(args.area, broad=args.broad)

    print("[INFO] Area:", args.area, "(broad)" if args.broad else "(anchored)")
    print("[INFO] Query:", q)
    print(f"[INFO] Filing date window: {from_date} to {to_date}")
    print(f"[INFO] Master rows: {len(existing)}")
    print(f"[INFO] Pulling next {args.add} starting at offset={start_offset}")

    client = PatentDataClient(api_key=api_key)

    pulled = fetch_next_batch(
    client,
    q=q,
    from_date=from_date,
    to_date=to_date,
    start_offset=start_offset,
    add=args.add,
    page_size=args.page_size,
)

    print(f"[INFO] Pulled {len(pulled)} records from ODP")

    # Append/dedupe by ip_number
    new_added = 0
    for app in pulled:
        ip_number, title, filing_date, abstract, inventors, assignees, contact_details = extract_fields(app)
        if not ip_number:
            continue

        r = Row(
            ip_number=ip_number,
            title=title,
            filing_date=filing_date,
            inventors=inventors,
            assignees=assignees,
            contact_details=contact_details,
            abstract=abstract,
            score=0,
            why_scored="",
            area=args.area,
        )

        if ip_number not in existing:
            existing[ip_number] = r
            new_added += 1
        else:
            # If new pull has richer fields, merge them in
            cur = existing[ip_number]
            cur.title = cur.title or r.title
            cur.filing_date = cur.filing_date or r.filing_date
            cur.inventors = cur.inventors or r.inventors
            cur.assignees = cur.assignees or r.assignees
            cur.contact_details = cur.contact_details or r.contact_details
            cur.abstract = cur.abstract or r.abstract
            # Keep most recent area tag (or you could preserve original)
            cur.area = cur.area or r.area

    print(f"[INFO] New unique added: {new_added}")

    # Rescore entire dataset
    all_rows = list(existing.values())
    for r in all_rows:
        text = f"{r.title} {r.abstract} {r.assignees} {r.inventors}"
        s, why = score_text(text)
        r.score = s
        r.why_scored = why

    all_rows.sort(key=lambda x: x.score, reverse=True)
    top_rows = all_rows[: min(args.top, len(all_rows))]

    # Write outputs (master + snapshot + top)
    out_dir = master_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = out_dir / f"odp_master_snapshot_{args.area}_{ts}.csv"
    top_path = out_dir / f"odp_top_{min(args.top, len(all_rows))}_{args.area}_{ts}.csv"

    try:
        write_csv(master_path, all_rows)
        print(f"[DONE] Updated master: {master_path}")
    except PermissionError:
        print(f"[WARN] Could not overwrite master (file locked). Writing snapshot only.")

    write_csv(snapshot_path, all_rows)
    write_csv(top_path, top_rows)

    print(f"[DONE] Snapshot: {snapshot_path}")
    print(f"[DONE] Top file: {top_path}")
    print(f"[INFO] Total master rows now: {len(all_rows)}")


if __name__ == "__main__":
    main()
