"""
이미지에 한글 쓰기

OpenCV의 글자 넣기 기능(cv2.putText)은 영어만 되고 한글은 ??? 로 깨진다.
그래서 Pillow(파이썬 이미지 라이브러리)로 맥에 기본 설치된 한글 폰트를 써서
글자를 그린 뒤 다시 OpenCV 이미지로 돌려준다.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 맥 기본 한글 폰트
FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]

_font_cache = {}


def get_font(size):
    """폰트를 크기별로 한 번만 읽어서 재사용한다 (매 프레임 읽으면 느리다)."""
    if size in _font_cache:
        return _font_cache[size]
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
            except Exception:
                continue
            _font_cache[size] = font
            return font
    # 한글 폰트를 못 찾으면 기본 폰트 (한글은 깨지지만 죽지는 않게)
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def draw_text(image_bgr, items):
    """
    이미지에 글자 여러 줄을 그린다.

    items: [(글자, (x, y), 크기, (B, G, R)), ...]
    반환: 글자가 그려진 새 이미지 (원본은 건드리지 않음)
    """
    pil = Image.fromarray(image_bgr[:, :, ::-1])   # BGR → RGB
    draw = ImageDraw.Draw(pil)
    for text, (x, y), size, color in items:
        draw.text((x, y), text, font=get_font(size),
                  fill=(color[2], color[1], color[0]))   # BGR → RGB
    return np.asarray(pil)[:, :, ::-1].copy()           # RGB → BGR
