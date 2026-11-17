"""Re-encode images so Ollama's vision path sees JPEG or PNG, not WebP or CMYK."""

import base64
import io
import unittest

from PIL import Image

from app.guard_image import image_b64, prepare_guard_image


def _jpeg_rgb(size=(32, 24), color=(12, 80, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _webp(size=(40, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 40, 40)).save(buf, "WEBP")
    return buf.getvalue()


def _cmyk_jpeg(size=(36, 28)) -> bytes:
    buf = io.BytesIO()
    Image.new("CMYK", size, (10, 20, 30, 40)).save(buf, "JPEG")
    return buf.getvalue()


class PrepareImageTests(unittest.TestCase):
    def test_a_plain_jpeg_is_left_alone(self) -> None:
        data = _jpeg_rgb()
        self.assertEqual(prepare_guard_image(data), data)

    def test_webp_becomes_a_jpeg_ollama_will_open(self) -> None:
        out = prepare_guard_image(_webp())
        with Image.open(io.BytesIO(out)) as im:
            self.assertEqual(im.format, "JPEG")
            self.assertEqual(im.mode, "RGB")

    def test_cmyk_jpeg_is_rewritten_as_rgb(self) -> None:
        out = prepare_guard_image(_cmyk_jpeg())
        with Image.open(io.BytesIO(out)) as im:
            self.assertEqual(im.format, "JPEG")
            self.assertEqual(im.mode, "RGB")

    def test_garbage_is_a_400_not_an_ollama_500(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            prepare_guard_image(b"not-an-image")
        self.assertIn("decoded", str(ctx.exception))

    def _huge_png(self, width: int, height: int) -> bytes:
        # A 1-bit blank PNG: tiny on the wire, huge once decoded.
        buf = io.BytesIO()
        Image.new("1", (width, height)).save(buf, "PNG")
        return buf.getvalue()

    def test_claimed_dimensions_over_budget_are_refused_before_decoding(self) -> None:
        data = self._huge_png(10_000, 10_000)  # 100 MP: only a warning to Pillow
        self.assertLess(len(data), 1024 * 1024)
        with self.assertRaises(ValueError):
            prepare_guard_image(data)

    def test_a_decompression_bomb_is_a_value_error(self) -> None:
        data = self._huge_png(20_000, 10_000)  # 200 MP: Pillow's DecompressionBombError
        with self.assertRaises(ValueError):
            prepare_guard_image(data)

    def test_a_large_jpeg_is_shrunk_to_the_edge_cap(self) -> None:
        buf = io.BytesIO()
        Image.new("RGB", (4000, 3000), (10, 120, 200)).save(buf, "JPEG")
        with Image.open(io.BytesIO(prepare_guard_image(buf.getvalue()))) as out:
            self.assertEqual(out.format, "JPEG")
            self.assertLessEqual(max(out.size), 1536)

    def test_image_b64_reencodes_a_data_url(self) -> None:
        raw = _webp()
        payload = "data:image/webp;base64," + base64.b64encode(raw).decode("ascii")
        decoded = base64.b64decode(image_b64(payload))
        with Image.open(io.BytesIO(decoded)) as im:
            self.assertEqual(im.format, "JPEG")

    def test_no_image_is_an_empty_string(self) -> None:
        self.assertEqual(image_b64(""), "")
        with self.assertRaises(ValueError):
            image_b64("@@@")

    def test_a_rotated_jpeg_is_transposed_and_its_exif_dropped(self) -> None:
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 CW on display
        buf = io.BytesIO()
        Image.new("RGB", (40, 20), (12, 80, 160)).save(buf, "JPEG", exif=exif.tobytes())
        with Image.open(io.BytesIO(prepare_guard_image(buf.getvalue()))) as out:
            self.assertEqual(out.size, (20, 40))
            self.assertNotIn(0x0112, out.getexif())

    def test_a_pa_image_is_saved_as_an_rgba_png(self) -> None:
        im = Image.new("PA", (16, 16))  # TIFF can carry PA; the PNG encoder cannot write it
        im.putpalette([200, 0, 0] * 256)
        with Image.open(io.BytesIO(prepare_guard_image(_pa_tiff(im)))) as out:
            self.assertEqual(out.format, "PNG")
            self.assertEqual(out.mode, "RGBA")


def _pa_tiff(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "TIFF")
    return buf.getvalue()
