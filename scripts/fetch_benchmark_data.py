"""Download and normalize the reviewed public benchmark CSVs.

This script deliberately has no dependency on DataPilot's runtime planner.
It makes benchmark data acquisition reproducible, records hashes, and keeps
the public source archive outside production query paths.
"""

from __future__ import annotations

import argparse
import io
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "benchmark_sources.json"
OUTPUT_DIR = ROOT / "data" / "real_benchmarks"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tabular_member(member, spec: dict) -> pd.DataFrame:
    """Read a manifest-declared CSV or Excel member without dataset-specific branches."""
    reader = spec.get("reader", "csv")
    options = spec.get("read_options", spec.get("read_csv", {}))
    if reader == "csv":
        return pd.read_csv(member, **options)
    if reader == "excel":
        return pd.read_excel(member, **options)
    raise ValueError(f"Unsupported benchmark reader '{reader}' for {spec.get('id')!r}.")


def fetch_dataset(spec: dict, *, force: bool) -> dict:
    output_path = OUTPUT_DIR / spec["output_file"]
    if output_path.exists() and not force:
        # Provenance is a schema contract for corpus validation.  Do not drop
        # rows/columns merely because a prior reproducible fetch already left
        # the normalized CSV on disk.
        existing = pd.read_csv(output_path)
        return {
            "id": spec["id"],
            "status": "already_present",
            "file": str(output_path.relative_to(ROOT)),
            "rows": int(len(existing)),
            "columns": list(existing.columns),
            "sha256": sha256_file(output_path),
        }

    archive_path = OUTPUT_DIR / f"{spec['id']}.zip"
    request = urllib.request.Request(
        spec["source_url"],
        headers={"User-Agent": "DataPilot-OS-benchmark-fetcher/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as output:
        shutil.copyfileobj(response, output)

    with zipfile.ZipFile(archive_path) as archive:
        try:
            # A public provider may put a versioned source archive inside a
            # catalogue ZIP. ``outer.zip::table.csv`` is a declarative source
            # manifest path, not a dataset-specific code branch.
            archive_member = spec["archive_member"]
            if "::" in archive_member:
                nested_member, csv_member = archive_member.split("::", 1)
                with archive.open(nested_member) as nested:
                    with zipfile.ZipFile(io.BytesIO(nested.read())) as nested_archive:
                        with nested_archive.open(csv_member) as member:
                            dataframe = read_tabular_member(member, spec)
            else:
                with archive.open(archive_member) as member:
                    dataframe = read_tabular_member(member, spec)
        except KeyError as exc:
            available = ", ".join(archive.namelist())
            raise RuntimeError(
                f"Source archive for {spec['id']} did not contain {spec['archive_member']!r}. "
                f"Available members: {available}"
            ) from exc
    archive_path.unlink(missing_ok=True)

    if dataframe.empty or not len(dataframe.columns):
        raise RuntimeError(f"Normalized benchmark {spec['id']} is empty or has no columns.")
    dataframe.to_csv(output_path, index=False)
    return {
        "id": spec["id"],
        "status": "downloaded",
        "file": str(output_path.relative_to(ROOT)),
        "rows": int(len(dataframe)),
        "columns": list(dataframe.columns),
        "sha256": sha256_file(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch reviewed DataPilot benchmark datasets.")
    parser.add_argument("--force", action="store_true", help="Replace an already downloaded normalized CSV.")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [fetch_dataset(spec, force=args.force) for spec in manifest["datasets"]]
    provenance_path = OUTPUT_DIR / "provenance.json"
    provenance_path.write_text(
        json.dumps({"manifest": str(MANIFEST.relative_to(ROOT)), "datasets": records}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
