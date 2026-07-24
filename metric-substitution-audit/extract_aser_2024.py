"""Extract the state-level access/learning panel from the ASER 2024 report.

Usage:
    python extract_aser_2024.py /path/to/ASER_2024_Final_Report.pdf

The extractor uses table structure rather than fixed page numbers:
- the first five-value row on each state enrollment page is the age 6-14 row;
- the third six-value row on the following reading page is the Std III row.

ASER's reading table reports mutually exclusive levels. The fifth value in the
Std III row is the percentage of all assessed children who can read a Std II
level text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

from pypdf import PdfReader


SOURCE_URL = (
    "https://asercentre.org/wp-content/uploads/2022/12/"
    "ASER_2024_Final-Report_13_2_24-1.pdf"
)
STATE_RE = re.compile(r"([A-Za-z &]+) RURAL")
ENROLLMENT_ROW_RE = re.compile(
    r"(?m)^(\d+\.\d) (\d+\.\d) (\d+\.\d) (\d+\.\d) 100$"
)
READING_ROW_RE = re.compile(
    r"(?m)^(\d+\.\d) (\d+\.\d) (\d+\.\d) "
    r"(\d+\.\d) (\d+\.\d) 100$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(pdf_path: Path) -> list[dict[str, object]]:
    reader = PdfReader(str(pdf_path))
    records: list[dict[str, object]] = []

    for page_index, page in enumerate(reader.pages[:-1]):
        text = page.extract_text() or ""
        if "School enrollment" not in text:
            continue

        state_match = STATE_RE.search(text)
        if not state_match:
            continue
        state = state_match.group(1).strip()
        if state == "India":
            continue

        enrollment_rows = ENROLLMENT_ROW_RE.findall(text)
        reading_text = reader.pages[page_index + 1].extract_text() or ""
        reading_rows = READING_ROW_RE.findall(reading_text)

        if len(enrollment_rows) < 1 or len(reading_rows) < 3:
            raise ValueError(
                f"Could not identify expected ASER rows near PDF page "
                f"{page_index + 1}."
            )

        govt, private, other, not_in_school = map(float, enrollment_rows[0])
        _, _, _, _, std_ii_reader = map(float, reading_rows[2])
        enrollment = round(100.0 - not_in_school, 1)

        records.append(
            {
                "state": state,
                "enrollment_age_6_14_pct": enrollment,
                "not_in_school_age_6_14_pct": not_in_school,
                "std3_read_std2_text_pct": std_ii_reader,
                "access_capability_wedge_pp": round(
                    enrollment - std_ii_reader, 1
                ),
                "govt_school_share_pct": govt,
                "private_school_share_pct": private,
                "other_school_share_pct": other,
                "source_pdf_page_index": page_index + 1,
                "source_url": SOURCE_URL,
            }
        )

    if len(records) != 27:
        raise ValueError(f"Expected 27 state/UT rows; extracted {len(records)}.")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "data"
        / "aser_2024_state_proxy_capability.csv",
    )
    args = parser.parse_args()

    records = extract(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Extracted {len(records)} state/UT rows.")
    print(f"Source SHA-256: {sha256(args.pdf)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
