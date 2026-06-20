#!/usr/bin/env python3
"""Rename 'Supplier Part' properties to 'LCSC' in KiCad schematic files.

Usage: supplier_to_lcsc.py <project_dir>

Processes all .kicad_sch files in the given directory (non-recursive).
Creates .bak backups before modifying.
"""

import re
import sys
import shutil
from pathlib import Path


def convert_file(path: Path) -> int:
    """Replace 'Supplier Part' with 'LCSC' in a schematic file. Returns count of replacements."""
    text = path.read_text()
    new_text, count = re.subn(
        r'(\(property\s+)"Supplier [Pp]art"',
        r'\1"LCSC"',
        text,
    )
    if count > 0:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text(new_text)
        print(f"  {path.name}: {count} properties renamed")
    else:
        print(f"  {path.name}: no 'Supplier Part' fields found")
    return count


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <project_dir>")
        sys.exit(1)

    project_dir = Path(sys.argv[1])
    if not project_dir.is_dir():
        print(f"Error: {project_dir} is not a directory")
        sys.exit(1)

    sch_files = sorted(project_dir.glob("*.kicad_sch"))
    if not sch_files:
        print(f"No .kicad_sch files found in {project_dir}")
        sys.exit(1)

    total = 0
    for f in sch_files:
        total += convert_file(f)

    print(f"\nDone. {total} total properties renamed to 'LCSC'.")


if __name__ == "__main__":
    main()
