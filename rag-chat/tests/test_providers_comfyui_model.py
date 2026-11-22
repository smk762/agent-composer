"""ComfyUI only loads the checkpoints it maps: an unknown model is a 400, never a raw ckpt_name."""

import asyncio
import unittest
from unittest import mock

from app.providers import comfyui


# pack-agent-composer-8b2fd61c83
class ComfyUIModelTests(unittest.TestCase):
    def test_unknown_model_is_refused_before_queueing(self) -> None:
        provider = comfyui.ComfyUIProvider(comfyui_url="http://comfy")
        queue = mock.AsyncMock(return_value="p1")
        with mock.patch.object(provider, "_queue_prompt", queue):
            with self.assertRaises(LookupError):
                asyncio.run(provider.generate_image("a cat", model="secret_ungated.safetensors"))
        queue.assert_not_called()

    def test_known_model_maps_to_its_checkpoint(self) -> None:
        provider = comfyui.ComfyUIProvider(comfyui_url="http://comfy")
        workflow = provider._build_txt2img_workflow("a", "", 512, 512, "flux-dev", seed=1)
        self.assertEqual(workflow["1"]["inputs"]["ckpt_name"], "flux1-dev.safetensors")


if __name__ == "__main__":
    unittest.main()
