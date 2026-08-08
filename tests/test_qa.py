import io

from PIL import Image

from cua_jepa.qa import changed_pixel_fraction, image_metrics, png_to_lossless_webp


def make_png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (20, 10), color).save(output, format="PNG")
    return output.getvalue()


def test_changed_pixel_fraction() -> None:
    black = make_png((0, 0, 0))
    white = make_png((255, 255, 255))
    assert changed_pixel_fraction(black, black) == 0.0
    assert changed_pixel_fraction(black, white) == 1.0


def test_webp_conversion_preserves_dimensions() -> None:
    original = make_png((12, 34, 56))
    webp = png_to_lossless_webp(original)
    with Image.open(io.BytesIO(webp)) as image:
        assert image.size == (20, 10)
    assert image_metrics(original).width == 20

