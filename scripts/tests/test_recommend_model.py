"""scripts/recommend_model.py: chat-model tiers. Stdlib only:

    python3 -m unittest discover -s scripts/tests
"""

import importlib.util
import sys
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "recommend_model", Path(__file__).resolve().parent.parent / "recommend_model.py")
recommend_model = importlib.util.module_from_spec(_SPEC)
sys.modules["recommend_model"] = recommend_model  # @dataclass looks its module up
_SPEC.loader.exec_module(recommend_model)


def _res(vram: float, ram: float = 16.0) -> "recommend_model.Resources":
    return recommend_model.Resources(cpu_cores=8, ram_gb=ram, gpu_vram_gb=vram, gpu_name="test")


class ChatModelTests(unittest.TestCase):
    def test_24gb_card_is_not_told_to_pull_70b(self) -> None:
        for mib in (24564, 24576):  # RTX 4090 reports 24564 MiB (~23.99 GiB)
            self.assertEqual(recommend_model.recommend_chat_model(_res(mib / 1024)), "llama3.1:8b")

    def test_48gb_card_gets_70b(self) -> None:
        self.assertEqual(recommend_model.recommend_chat_model(_res(49140 / 1024)), "llama3.1:70b")


if __name__ == "__main__":
    unittest.main()
