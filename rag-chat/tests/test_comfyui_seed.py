"""ComfyUI txt2img workflows carry a valid KSampler seed. No network."""

import unittest

from app.providers import comfyui


class ComfyUISeedTests(unittest.TestCase):
    def test_seed_is_non_negative_and_varies(self) -> None:
        provider = comfyui.ComfyUIProvider()
        seeds = {
            provider._build_txt2img_workflow("p", "", 512, 512, "flux-dev")["5"]["inputs"]["seed"]
            for _ in range(5)
        }
        self.assertTrue(all(isinstance(s, int) and s >= 0 for s in seeds))
        self.assertGreater(len(seeds), 1)

    def test_explicit_seed_is_kept(self) -> None:
        wf = comfyui.ComfyUIProvider()._build_txt2img_workflow("p", "", 512, 512, "flux-dev", seed=42)
        self.assertEqual(wf["5"]["inputs"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
