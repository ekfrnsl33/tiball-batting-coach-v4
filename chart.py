"""
간단한 꺾은선 그래프 그리기

matplotlib을 쓰지 않고 OpenCV로 직접 그린다. 결과 이미지를 그대로
Flask 화면(5단계)에도 쓸 수 있게 하려는 목적이다.
"""

import cv2
import numpy as np

from textdraw import draw_text

BG = (28, 28, 28)
AXIS = (90, 90, 90)
TEXT = (225, 225, 225)
MARK = (60, 60, 235)

PALETTE = [(240, 200, 40), (80, 220, 120), (230, 140, 230)]


def draw_series(series, peak_index=None, width=900, height=280,
                title=None, x_label=None):
    """
    series: {이름: 값 배열(nan 허용)}
    peak_index: 세로선으로 표시할 프레임 번호
    """
    image = np.full((height, width, 3), BG, dtype=np.uint8)
    left, right, top, bottom = 60, width - 16, 46, height - 40
    cv2.rectangle(image, (left, top), (right, bottom), AXIS, 1)

    # 단위가 다른 값들(미터/초 vs 픽셀/초)을 함께 그리려면 각 곡선을
    # 자기 최대값 기준으로 맞춰야 한다. 안 그러면 작은 쪽이 바닥에 붙어
    # 아예 보이지 않는다. 비교하려는 것은 봉우리의 '위치와 모양'이다.
    scales = {}
    for name, values in series.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        scales[name] = float(finite.max()) if finite.size and finite.max() > 0 \
            else 1.0
    if not scales:
        return image
    length = max(len(v) for v in series.values())

    def px(i):
        return int(left + (right - left) * i / max(length - 1, 1))

    def py(v, name):
        return int(bottom - (bottom - top) * (v / scales[name]) * 0.94)

    # 봉우리 위치 세로선
    if peak_index is not None:
        cv2.line(image, (px(peak_index), top), (px(peak_index), bottom),
                 MARK, 2)

    for color, (name, values) in zip(PALETTE, series.items()):
        values = np.asarray(values, dtype=float)
        previous = None
        for i, value in enumerate(values):
            if not np.isfinite(value):
                previous = None
                continue
            point = (px(i), py(value, name))
            if previous is not None:
                cv2.line(image, previous, point, color, 2)
            previous = point

    items = []
    if title:
        items.append((title, (14, 10), 22, TEXT))
    # 곡선마다 최대값이 다르므로 범례에 각자의 최대값을 함께 적는다
    for i, (color, name) in enumerate(zip(PALETTE, series)):
        items.append((f"— {name} (최대 {scales[name]:.1f})",
                      (14 + i * 250, height - 26), 17, color))
    items.append(("0", (10, bottom - 16), 16, TEXT))
    items.append(("각 곡선을 자기 최대값 기준으로 맞춤",
                  (width - 250, 12), 16, AXIS))
    if x_label:
        items.append((x_label, (right - 60, bottom + 6), 17, TEXT))
    if peak_index is not None:
        items.append((f"임팩트 {peak_index}", (min(px(peak_index) + 6,
                                                width - 130), top + 4),
                      17, MARK))
    return draw_text(image, items)
