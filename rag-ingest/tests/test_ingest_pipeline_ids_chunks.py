"""Point ids and text chunk windows (ingest_pipeline._point_id / chunk_text)."""

import hashlib
import os
import tempfile
import unittest
import uuid

os.environ.setdefault("INGEST_NONCE_DB", os.path.join(tempfile.mkdtemp(prefix="ingest-test-"), "nonces.db"))

from app import ingest_pipeline as ip


class PointIdTests(unittest.TestCase):
    def test_pipe_in_source_cannot_shift_the_field_boundary(self) -> None:
        self.assertNotEqual(ip._point_id("a|b", "c", 0, "x"), ip._point_id("a", "b|c", 0, "x"))

    def test_ids_without_a_pipe_in_source_are_unchanged(self) -> None:
        # Stored points keep their ids, so a re-send still overwrites them in place.
        digest = hashlib.sha256(b"x").hexdigest()
        self.assertEqual(ip._point_id("s", "d|e", 3, "x"),
                         str(uuid.uuid5(uuid.NAMESPACE_URL, f"s|d|e|3|{digest}")))
        self.assertEqual(ip._point_id(None, None, 0, "x"),
                         str(uuid.uuid5(uuid.NAMESPACE_URL, f"||0|{digest}")))


class ChunkTextTests(unittest.TestCase):
    def test_no_tail_chunk_contained_in_the_previous_one(self) -> None:
        t = "".join(chr(97 + i % 26) for i in range(700))
        self.assertEqual(ip.chunk_text(t, chunk_size=800, overlap=200), [t])

    def test_windows_still_cover_the_whole_text(self) -> None:
        t = "".join(chr(97 + i % 26) for i in range(1500))
        chunks = ip.chunk_text(t, chunk_size=800, overlap=200)
        self.assertEqual(chunks, [t[0:800], t[600:1400], t[1200:1500]])

    def test_large_overlap_emits_no_repeated_tails(self) -> None:
        t = "".join(chr(97 + i % 26) for i in range(100))
        chunks = ip.chunk_text(t, chunk_size=50, overlap=40)
        self.assertEqual(chunks, [t[i:i + 50] for i in range(0, 60, 10)])  # stops at the window ending at 100


if __name__ == "__main__":
    unittest.main()
