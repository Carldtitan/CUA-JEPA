from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageStat
from playwright.sync_api import Page


@dataclass(frozen=True)
class ImageMetrics:
    sha256: str
    width: int
    height: int
    luminance_stddev: float


def image_metrics(png: bytes) -> ImageMetrics:
    with Image.open(io.BytesIO(png)) as image:
        gray = image.convert("L")
        return ImageMetrics(
            sha256=hashlib.sha256(png).hexdigest(),
            width=image.width,
            height=image.height,
            luminance_stddev=float(ImageStat.Stat(gray).stddev[0]),
        )


def changed_pixel_fraction(before_png: bytes, after_png: bytes) -> float:
    with Image.open(io.BytesIO(before_png)).convert("RGB") as before:
        with Image.open(io.BytesIO(after_png)).convert("RGB") as after:
            if before.size != after.size:
                return 1.0
            diff = ImageChops.difference(before, after).convert("L")
            histogram = diff.histogram()
            changed = sum(histogram[1:])
            return changed / (before.width * before.height)


def png_to_lossless_webp(png: bytes) -> bytes:
    output = io.BytesIO()
    with Image.open(io.BytesIO(png)).convert("RGB") as image:
        image.save(output, format="WEBP", lossless=True, method=4)
    return output.getvalue()


def stable_screenshot(page: Page, attempts: int = 6, interval_ms: int = 250) -> bytes:
    previous = page.screenshot(type="png", animations="disabled", caret="hide")
    for _ in range(attempts):
        page.wait_for_timeout(interval_ms)
        current = page.screenshot(type="png", animations="disabled", caret="hide")
        if current == previous:
            return current
        previous = current
    return previous


def is_usable_screen(metrics: ImageMetrics) -> bool:
    return metrics.width >= 640 and metrics.height >= 480 and metrics.luminance_stddev >= 5.0

