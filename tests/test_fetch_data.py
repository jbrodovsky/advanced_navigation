"""Tests for scripts/fetch_data.py and scripts/package_datasets.py (no network access)."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


fetch_data = _load_script("fetch_data")
package_datasets = _load_script("package_datasets")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return path


def _source(url: str, sha256: str = "", **overrides) -> object:
    fields = dict(
        name="toy",
        description="toy dataset",
        provider="https",
        url=url,
        archive="toy.zip",
        expected_files=("a.txt", "sub/b.txt"),
        sha256=sha256,
    )
    fields.update(overrides)
    return fetch_data.DataSource(**fields)


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class NormalizeShareUrlTests(unittest.TestCase):
    def test_google_drive_file_links(self) -> None:
        expected = "https://drive.usercontent.google.com/download?id=AbC_123-x&export=download&confirm=t"
        for link in (
            "https://drive.google.com/file/d/AbC_123-x/view?usp=sharing",
            "https://drive.google.com/open?id=AbC_123-x",
            "https://drive.google.com/uc?id=AbC_123-x&export=download",
        ):
            with self.subTest(link=link):
                self.assertEqual(fetch_data.normalize_share_url(link), [expected])

    def test_google_drive_folder_link_is_rejected(self) -> None:
        with self.assertRaises(fetch_data.FetchError):
            fetch_data.normalize_share_url("https://drive.google.com/drive/folders/XYZ?usp=sharing")

    def test_sharepoint_link_gets_download_flag(self) -> None:
        link = "https://wpi0-my.sharepoint.com/:u:/g/personal/x/EaBc?e=AbCd"
        (candidate,) = fetch_data.normalize_share_url(link)
        self.assertIn("download=1", candidate)
        self.assertIn("e=AbCd", candidate)

    def test_personal_onedrive_link_has_api_fallback(self) -> None:
        candidates = fetch_data.normalize_share_url("https://1drv.ms/u/s!AbCdEf")
        self.assertEqual(len(candidates), 2)
        self.assertTrue(candidates[1].startswith("https://api.onedrive.com/v1.0/shares/u!"))

    def test_plain_url_is_unchanged(self) -> None:
        self.assertEqual(fetch_data.normalize_share_url("https://example.org/x.zip"), ["https://example.org/x.zip"])


class FetchDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.data_dir = self.tmp / "data"
        self.archive = _make_zip(
            self.tmp / "toy.zip",
            {"toy/a.txt": b"alpha", "toy/sub/b.txt": b"beta"},
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_download_extracts_and_strips_top_folder(self) -> None:
        source = _source(self.archive.as_uri(), _sha256(self.archive))
        status = _quiet(fetch_data.fetch_dataset, source, self.data_dir)
        self.assertEqual(status, fetch_data.EXIT_OK)
        target = self.data_dir / "toy"
        self.assertEqual((target / "a.txt").read_text(), "alpha")
        self.assertEqual((target / "sub" / "b.txt").read_text(), "beta")
        self.assertTrue((target / fetch_data.STAMP_NAME).exists())
        self.assertFalse((self.data_dir / ".downloads" / "toy.zip").exists())

    def test_second_run_skips(self) -> None:
        source = _source(self.archive.as_uri(), _sha256(self.archive))
        _quiet(fetch_data.fetch_dataset, source, self.data_dir)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            fetch_data.fetch_dataset(source, self.data_dir)
        self.assertIn("already present", output.getvalue())

    def test_checksum_mismatch_leaves_no_dataset(self) -> None:
        source = _source(self.archive.as_uri(), "0" * 64)
        with self.assertRaises(fetch_data.ChecksumMismatchError):
            _quiet(fetch_data.fetch_dataset, source, self.data_dir)
        self.assertFalse((self.data_dir / "toy").exists())

    def test_html_response_is_rejected(self) -> None:
        page = self.tmp / "page.html"
        page.write_text("<html><body>Sign in</body></html>")
        with self.assertRaises(fetch_data.NotAnArchiveError):
            _quiet(fetch_data.fetch_dataset, _source(page.as_uri()), self.data_dir)

    def test_placeholder_url_returns_exit_code(self) -> None:
        status = _quiet(fetch_data.fetch_dataset, _source("REPLACE_ME", provider="auto"), self.data_dir)
        self.assertEqual(status, fetch_data.EXIT_PLACEHOLDER)

    def test_manual_archive_install(self) -> None:
        source = _source("REPLACE_ME", _sha256(self.archive), provider="auto")
        status = _quiet(fetch_data.fetch_dataset, source, self.data_dir, archive=self.archive)
        self.assertEqual(status, fetch_data.EXIT_OK)
        self.assertTrue((self.data_dir / "toy" / "a.txt").exists())

    def test_missing_expected_file_is_an_error(self) -> None:
        source = _source(self.archive.as_uri(), expected_files=("a.txt", "missing.txt"))
        with self.assertRaises(fetch_data.FetchError):
            _quiet(fetch_data.fetch_dataset, source, self.data_dir)

    def test_unsafe_member_is_rejected(self) -> None:
        evil = _make_zip(self.tmp / "evil.zip", {"../escape.txt": b"x"})
        with self.assertRaises(fetch_data.FetchError):
            fetch_data.safe_extract(evil, self.data_dir / "toy")
        self.assertFalse((self.data_dir / "escape.txt").exists())


class DataSourcesFileTests(unittest.TestCase):
    def test_sources_file_lists_the_three_course_datasets(self) -> None:
        sources = fetch_data.load_sources()
        self.assertEqual(set(sources), {"drone-data", "victoria-park", "mems-nav-dataset"})
        for source in sources.values():
            self.assertTrue(source.expected_files, source.name)
            if source.provider != "github":
                self.assertEqual(len(source.sha256), 64, source.name)
                self.assertIsNotNone(source.issue, source.name)


class PackageDatasetsTests(unittest.TestCase):
    def test_archives_are_deterministic_and_installable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_dir = tmp / "src"
            source_dir.mkdir()
            (source_dir / "a.txt").write_text("alpha")
            (source_dir / "b.bin").write_bytes(bytes(range(256)))

            first = package_datasets.build_archive("toy", source_dir, ["a.txt", "b.bin"], tmp / "out1")
            second = package_datasets.build_archive("toy", source_dir, ["a.txt", "b.bin"], tmp / "out2")
            self.assertEqual(_sha256(first), _sha256(second))

            source = _source(first.as_uri(), _sha256(first), expected_files=("a.txt", "b.bin"))
            _quiet(fetch_data.fetch_dataset, source, tmp / "data")
            self.assertEqual((tmp / "data" / "toy" / "b.bin").read_bytes(), bytes(range(256)))

    def test_missing_source_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                package_datasets.build_archive("toy", Path(tmp_dir), ["nope.txt"], Path(tmp_dir) / "out")


if __name__ == "__main__":
    unittest.main()
