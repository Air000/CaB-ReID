import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/verify_checkpoints.py"


class VerifyCheckpointsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.names = ("clip_region_adapter.pt", "cabreid_region_encoder.pth", "cabreid_clipreid_mvti.pth")
        rows = []
        for name in self.names:
            content = name.encode()
            (self.root / name).write_bytes(content)
            rows.append(f"{hashlib.sha256(content).hexdigest()}  {name}\n")
        (self.root / "CHECKSUMS.sha256").write_text("".join(rows))

    def verify(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--weights-dir", str(self.root),
             "--dataset", "mvti", "--backbone", "clipreid"],
            capture_output=True, text=True,
        )

    def test_single_model_download(self):
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in self.names:
            self.assertIn(f"OK: {name}", result.stdout)

    def test_shared_encoder_is_always_verified(self):
        (self.root / "cabreid_region_encoder.pth").write_bytes(b"corrupt")
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checksum mismatch: cabreid_region_encoder.pth", result.stderr)

    def test_incomplete_manifest_is_rejected(self):
        (self.root / "CHECKSUMS.sha256").write_text("")
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checksum manifest is missing", result.stderr)
