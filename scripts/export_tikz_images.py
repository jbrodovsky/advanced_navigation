"""Export standalone TikZ sources under book/images/src to PNG.

This script keeps LaTeX build byproducts in a temporary directory, then writes
only the final PNG files into book/images.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "book" / "images" / "src"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "book" / "images"


def require_tool(name: str) -> str:
    """Return the executable path for a required command-line tool."""
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required tool not found on PATH: {name}")
    return path


def run(command: list[str]) -> None:
    """Run a command and show a compact error if it fails."""
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(command)}")


def export_tikz(
    source: Path,
    output_dir: Path,
    density: int,
    pdflatex: str,
    magick: str,
) -> Path:
    """Compile one standalone TikZ source and copy its PNG to the output dir."""
    source = source.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    png_target = output_dir / f"{source.stem}.png"

    with tempfile.TemporaryDirectory(prefix=f"{source.stem}-tikz-") as tmp:
        tmp_dir = Path(tmp)
        generated_pdf = tmp_dir / f"{source.stem}.pdf"
        generated_png = tmp_dir / f"{source.stem}.png"

        run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={tmp_dir}",
                str(source),
            ]
        )

        generated_pdf = tmp_dir / f"{source.stem}.pdf"
        if not generated_pdf.exists():
            raise RuntimeError(f"expected PDF was not created: {generated_pdf}")

        run(
            [
                magick,
                "-density",
                str(density),
                str(generated_pdf),
                "-quality",
                "95",
                str(generated_png),
            ]
        )

        if not generated_png.exists():
            raise RuntimeError(f"expected PNG was not created: {generated_png}")

        shutil.copy2(generated_png, png_target)

    return png_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export standalone TikZ .tex files to book-ready PNG files."
    )
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="Specific .tex sources to export. Defaults to book/images/src/*.tex.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported PNG files.",
    )
    parser.add_argument(
        "--density",
        type=int,
        default=300,
        help="PNG render density in DPI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        pdflatex = require_tool("pdflatex")
        magick = require_tool("magick")

        sources = args.sources or sorted(DEFAULT_SOURCE_DIR.glob("*.tex"))
        if not sources:
            raise RuntimeError(f"no TikZ sources found in {DEFAULT_SOURCE_DIR}")

        for source in sources:
            if source.suffix != ".tex":
                raise RuntimeError(f"expected a .tex source, got: {source}")
            png = export_tikz(source, args.output_dir, args.density, pdflatex, magick)
            print(f"exported {source} -> {png}")

    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
