"""scripts/ingest_code.py: file discovery under --root. Stdlib only:

    python3 -m unittest discover -s scripts/tests
"""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("ingest_code", Path(__file__).resolve().parent.parent / "ingest_code.py")
ingest_code = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ingest_code)


class SourceFilesTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="ingest-code-test-"))
        self.real = tmp / "real"
        (self.real / "src").mkdir(parents=True)
        (self.real / "src" / "a.py").write_text("x = 1\n")
        (self.real / "node_modules").mkdir()
        (self.real / "node_modules" / "b.py").write_text("y = 2\n")
        (tmp / "shared").mkdir()
        (tmp / "shared" / "c.py").write_text("z = 3\n")
        self.link = tmp / "link"
        self.link.symlink_to(self.real, target_is_directory=True)
        self.tmp = tmp

    def test_root_through_a_symlink(self) -> None:
        files = ingest_code.source_files(self.link, ["."], {".py"})
        self.assertEqual(files, [self.real.resolve() / "src" / "a.py"])

    def test_relative_root(self) -> None:
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            files = ingest_code.source_files(Path("real"), ["src"], {".py"})
        finally:
            os.chdir(cwd)
        self.assertEqual(files, [self.real.resolve() / "src" / "a.py"])

    def test_dir_outside_root_is_refused_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside --root"):
            ingest_code.source_files(self.real, ["../shared"], {".py"})

    def test_main_reports_outside_dir_without_a_traceback(self) -> None:
        self.assertEqual(ingest_code.main(["--root", str(self.real), "--dry-run", "../shared"]), 2)


class ArgsTests(unittest.TestCase):
    def test_exts_are_stripped_before_the_dot(self) -> None:
        self.assertEqual(ingest_code.parse_exts("py, md ,.ts,,"), {".py", ".md", ".ts"})

    def test_batch_below_one_is_rejected(self) -> None:
        for bad in ("0", "-3"):
            with self.subTest(batch=bad), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as cm:
                ingest_code.main(["--batch", bad, "--dry-run"])
            self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
