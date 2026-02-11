# USPTO ODP Patent Scraper (H4D / RapidRotor-AI)

A Python tool for incrementally scraping USPTO Open Data Portal (ODP) patent applications filtered by keyword areas (rotor wake, meshing, AI-assisted CFD, unsteady turbulence, etc.). Designed for landscape analysis and contact discovery in advanced rotorcraft design.

## Features

- **Incremental pulling**: Append new records to a master CSV with automatic deduplication by `ip_number`
- **Smart scoring**: Relevance-based ranking using CFD, aerodynamic, and domain-specific signals
- **Multiple keyword areas**: Pre-defined search term libraries for rotorcraft, meshing, optimization, AI/CFD, etc.
- **Flexible querying**: 
  - Anchored mode (requires CFD + aero context; high precision)
  - Broad mode (aero + area terms; wider net)
- **Timestamped outputs**: Master CSV + snapshot + top-N shortlist per run
- **File-lock safe**: Snapshots always succeed even if master is locked by Excel

## Output

Each run produces:
- **Master CSV**: All records found so far, rescored after each append
- **Snapshot CSV**: Timestamped copy of entire master (for backup/version control)
- **Top-N CSV**: Filtered to top scores (default 500 records)

### Output columns
| Column | Description |
|--------|-------------|
| `ip_number` | USPTO application/patent identifier |
| `title` | Invention title |
| `filing_date` | Application filing date (ISO format) |
| `inventors` | First inventor name (best-effort from ODP) |
| `assignees` | First applicant/assignee name |
| `contact_details` | Docket + customer numbers (sparse) |
| `abstract` | Full abstract text |
| `score` | Relevance score (0–50+ range) |
| `why_scored` | Matched keyword signals |
| `area` | Keyword area used in this pull |

## Prerequisites

- **Python**: 3.10+
- **USPTO ODP API Key**: Free tier available at https://developer.uspto.gov/

## Installation

```bash
git clone https://github.com/yourusername/h4d-webscraping.git
cd h4d-webscraping

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt