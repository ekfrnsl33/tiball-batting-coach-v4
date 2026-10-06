"""
미리보기 화면 그리기 (preview.py와 test_angle.py가 함께 사용)

설계 의도:
  촬영하는 사람이 카메라에서 2~3m 떨어져 서 있으므로, 맥북 화면의 작은
  글자는 읽을 수 없다. 그래서 멀리서도 알아볼 수 있는 신호를 우선한다.

    - 화면 전체 테두리 색  : 초록 = 준비 완료 / 노랑 = 조정 필요 / 빨강 = 촬영 중
    - 아주 큰 카운트다운 숫자
    - 큰 상태 글자

  자세한 수치(관절별 상태, 거리)는 아래 패널에 작게 넣는다. 이건 가까이
  앉아서 카메라 위치를 맞출 때 보는 정보다.
"""

import cv2
import numpy as np

from pose import LANDMARK_NAMES, VISIBILITY_THRESHOLD, draw_skeleton
from textdraw import draw_text

WINDOW = "Tee-ball Camera Setup"

# 안전 영역: 화면 가장자리에서 이 비율만큼 안쪽.
# 관절이 딱 경계에 걸치면 조금만 움직여도 밖으로 나가므로 여유를 둔다.
MARGIN_RATIO = 0.04

# 멀리서 보려면 창이 커야 한다 (640x480 → 1.5배 = 960x720)
DISPLAY_SCALE = 1.5
BORDER = 18          # 상태를 알려주는 테두리 두께
PANEL_HEIGHT = 112   # 아래쪽 상세 정보 패널

GREEN = (60, 200, 60)
RED = (60, 60, 235)
YELLOW = (40, 200, 240)
WHITE = (245, 245, 245)
GRAY = (150, 150, 150)


def check_framing(landmarks, joint_set, depth_image, cam, shape):
    """
    판정에 쓰는 6개 지점이 제대로 잡히는지 확인하고,
    어긋났으면 카메라를 어떻게 움직여야 하는지까지 정해서 돌려준다.
    """
    h, w = shape[:2]
    mx, my = int(w * MARGIN_RATIO), int(h * MARGIN_RATIO)

    rows = []
    all_ok = True
    max_y = min_y = None

    for role, idx in joint_set.items():
        lm = landmarks[idx]
        inside = (mx <= lm["x"] < w - mx) and (my <= lm["y"] < h - my)
        depth_m = cam.sample_depth(depth_image, lm["x"], lm["y"])
        visible = lm["visibility"] >= VISIBILITY_THRESHOLD
        ok = inside and depth_m is not None
        all_ok = all_ok and ok
        rows.append({
            "role": role,
            "name": LANDMARK_NAMES[idx],
            "inside": inside,
            "depth": depth_m,
            "visible": visible,
            "ok": ok,
        })
        max_y = lm["y"] if max_y is None else max(max_y, lm["y"])
        min_y = lm["y"] if min_y is None else min(min_y, lm["y"])

    advice = None
    below = max_y - (h - my)      # 아래로 얼마나 넘쳤는지
    above = my - min_y            # 위로 얼마나 넘쳤는지
    if below > 0 and above > 0:
        advice = "몸이 화면보다 큽니다 → 더 멀리 떨어져 주세요"
    elif below > 0:
        headroom = min_y - my
        if headroom > below:
            advice = "발이 잘림 → 카메라를 아래로 기울여 주세요"
        else:
            advice = "발이 잘림 → 뒤로 조금 물러나 주세요"
    elif above > 0:
        footroom = (h - my) - max_y
        if footroom > above:
            advice = "위가 잘림 → 카메라를 위로 세워 주세요"
        else:
            advice = "위가 잘림 → 뒤로 조금 물러나 주세요"
    elif not all_ok:
        missing = [r["name"] for r in rows if not r["ok"]]
        advice = f"측정 안 됨: {', '.join(missing)}"

    return rows, all_ok, advice


def _draw_safe_zone(image):
    h, w = image.shape[:2]
    mx, my = int(w * MARGIN_RATIO), int(h * MARGIN_RATIO)
    step = 22
    for x in range(mx, w - mx, step):
        cv2.line(image, (x, my), (min(x + 11, w - mx), my), GREEN, 2)
        cv2.line(image, (x, h - my), (min(x + 11, w - mx), h - my), GREEN, 2)
    for y in range(my, h - my, step):
        cv2.line(image, (mx, y), (mx, min(y + 11, h - my)), GREEN, 2)
        cv2.line(image, (w - mx, y), (w - mx, min(y + 11, h - my)), GREEN, 2)
    return image


def render(frame, landmarks, joint_set, cam, side, fps,
           guide_lines=None, countdown=None, state=None, footer=None):
    """
    보여줄 큰 화면을 만든다.

    state: None이면 준비 상태에 따라 자동(초록/노랑).
           "recording"이면 빨간 테두리 + '촬영 중'.
    반환: (합성 이미지, rows, all_ok, advice)
    """
    color = frame["color"]

    # 골격선·안전영역은 원본 크기에 그린 뒤 한 번에 확대한다
    view = color
    if landmarks is not None:
        view = draw_skeleton(view, landmarks,
                             highlight_indices=sorted(set(joint_set.values())))
    else:
        view = view.copy()
    view = _draw_safe_zone(view)

    if landmarks is not None:
        rows, all_ok, advice = check_framing(
            landmarks, joint_set, frame["depth"], cam, color.shape)
        hip = next((r["depth"] for r in rows if r["role"] == "hip"), None)
    else:
        rows, all_ok = [], False
        advice = "사람이 안 보입니다"
        hip = None

    big = cv2.resize(view, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                     interpolation=cv2.INTER_LINEAR)
    h, w = big.shape[:2]

    # -- 상태에 따른 테두리 (멀리서도 보이는 가장 중요한 신호) --------------
    if state == "recording":
        border_color, status_text = RED, "촬영 중!"
    elif all_ok:
        border_color, status_text = GREEN, "준비 완료"
    else:
        border_color, status_text = YELLOW, "조정 필요"
    cv2.rectangle(big, (0, 0), (w - 1, h - 1), border_color, BORDER * 2)

    items = []

    # 자세 안내문 — 위쪽에 반투명 띠를 깔고 크게
    if guide_lines:
        band_h = 46 * len(guide_lines) + 20
        y0 = BORDER
        strip = big[y0:y0 + band_h, BORDER:w - BORDER].astype(np.float32) * 0.2
        big[y0:y0 + band_h, BORDER:w - BORDER] = strip.astype(np.uint8)
        for i, line in enumerate(guide_lines):
            items.append((line, (BORDER + 14, y0 + 8 + i * 46), 34, WHITE))

    # 상태 글자 — 크게. 몸이 겹쳐도 읽히게 아래쪽에 어두운 띠를 깐다.
    band_top = h - 128
    strip = big[band_top:h - BORDER, BORDER:w - BORDER].astype(np.float32) * 0.25
    big[band_top:h - BORDER, BORDER:w - BORDER] = strip.astype(np.uint8)
    items.append((status_text, (BORDER + 14, band_top + 8), 46, border_color))
    if state != "recording":
        second = advice if advice else (
            f"거리 {hip:.2f} m — 좋습니다" if hip else None)
        if second:
            items.append((second, (BORDER + 14, band_top + 70), 30,
                          YELLOW if advice else GREEN))

    # 카운트다운 — 화면을 가로지를 만큼 크게
    if countdown is not None:
        items.append((str(countdown), (w - 240, h // 2 - 190), 300,
                       border_color))

    # -- 아래 상세 패널 (가까이서 보는 정보) --------------------------------
    panel = np.full((PANEL_HEIGHT, w, 3), 28, dtype=np.uint8)
    panel_items = [(f"{'우타자' if side == 'right' else '좌타자'}   "
                    f"{fps:.0f} fps", (14, 8), 20, GRAY)]
    for i, row in enumerate(rows):
        px = 14 + (i % 3) * 300
        py = 40 + (i // 3) * 32
        if not row["inside"]:
            mark, col = "화면밖", RED
        elif row["depth"] is None:
            mark, col = "심도X", RED
        elif not row["visible"]:
            # 화면 안에 있고 거리도 읽히지만 MediaPipe 확신도(γ)가 낮은 상태.
            # 몸에 가려졌을 때 생기며, 측정값 신뢰도가 떨어진다.
            mark, col = "가려짐", YELLOW
        else:
            mark, col = "O", GREEN
        dist = f"{row['depth']:.2f}m" if row["depth"] else "-"
        panel_items.append((f"{row['name']} {mark} {dist}", (px, py), 21, col))
    if footer:
        panel_items.append((footer, (w - 14 - 11 * len(footer), 8), 19, GRAY))

    composed = np.vstack([big, panel])
    items += [(t, (x, y + h), s, c) for t, (x, y), s, c in panel_items]
    return draw_text(composed, items), rows, all_ok, advice
