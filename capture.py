"""
촬영 + 기록 만들기 (3단계 이후 모든 단계에서 공용)

'기록(record)'이란 한 프레임에서 뽑아낸 것들을 묶은 것이다.

    {
      "x", "y", "v" : 관절 33개의 화면 좌표와 가시성
      "patches"     : 관절 33개 위치의 심도 조각 (11x11) — 나중에 거리로 환산
      "color"       : 그 프레임의 컬러 이미지
      "landmarks"   : pose.py가 준 원래 형식 (골격선 그릴 때 씀)
    }

심도를 미리 거리(m)로 바꿔두지 않고 조각째로 들고 있는 이유:
거리를 골라내는 방법이 여러 개이고(depth_sampling.py), 나중에 다른 방법으로
다시 계산해보려면 원본이 필요하기 때문이다.
"""

import time

import cv2
import numpy as np

from depth_sampling import PATCH_RADIUS
from preview_ui import WINDOW, render


SIDE_LABEL = {"left": "좌타자 (왼팔·왼다리)", "right": "우타자 (오른팔·오른다리)"}


def side_from_args(argv, default="left"):
    """
    실행할 때 좌타/우타를 고른다 (명세서 3번: 자동 인식 없이 사전 선택).

        sudo ./venv/bin/python test_impact.py --side right

    기본값이 중요하다: 좌타자를 우타자로 재면 뒷팔·뒷다리가 아니라
    앞팔·앞다리를 재게 되어 완전히 다른 값이 나온다.
    """
    for i, arg in enumerate(argv):
        if arg in ("--side", "-s") and i + 1 < len(argv):
            value = argv[i + 1].lower()
            if value in SIDE_LABEL:
                return value
        if arg == "--left":
            return "left"
        if arg == "--right":
            return "right"
    return default


def fixed_patch(cam, depth_image, x, y):
    """항상 같은 크기로 심도 조각을 잘라온다. 빈 곳은 0(=측정 실패)."""
    size = 2 * PATCH_RADIUS + 1
    out = np.zeros((size, size), dtype=np.uint16)
    patch = cam.depth_patch(depth_image, x, y, PATCH_RADIUS)
    if patch.size:
        out[:patch.shape[0], :patch.shape[1]] = patch
    return out


def build_records(cam, extractor, clip):
    """녹화된 프레임들에서 관절을 뽑아 기록 목록을 만든다."""
    records = []
    for frame in clip:
        landmarks = extractor.detect(frame["color"], frame["timestamp_ms"])
        if landmarks is None:
            continue
        records.append({
            "x": np.array([lm["x"] for lm in landmarks], dtype=np.float32),
            "y": np.array([lm["y"] for lm in landmarks], dtype=np.float32),
            "v": np.array([lm["visibility"] for lm in landmarks],
                          dtype=np.float32),
            "patches": np.stack([
                fixed_patch(cam, frame["depth"], lm["x"], lm["y"])
                for lm in landmarks]),
            "color": frame["color"],
            "landmarks": landmarks,
        })
    return records


def save_raw(path, records, cam):
    """다시 촬영하지 않고도 재분석할 수 있게 원본을 저장한다."""
    if not records:
        return
    intr = cam.intrinsics
    np.savez_compressed(
        path,
        x=np.stack([r["x"] for r in records]),
        y=np.stack([r["y"] for r in records]),
        visibility=np.stack([r["v"] for r in records]),
        patches=np.stack([r["patches"] for r in records]),
        intrinsics=np.array([intr.fx, intr.fy, intr.ppx, intr.ppy],
                            dtype=np.float32),
        depth_scale=np.float32(cam.depth_scale),
    )


def countdown_with_preview(cam, extractor, joint_set, side, seconds,
                           guide_lines):
    """
    카운트다운 동안 미리보기 창을 띄운다.
    멀리 서 있어도 알 수 있게 테두리 색과 큰 숫자로 표시한다.
    창에서 q나 esc를 누르면 중단(False 반환).
    """
    end = time.monotonic() + seconds
    last = time.monotonic()
    fps = 0.0
    while True:
        remain = end - time.monotonic()
        if remain <= 0:
            return True
        frame = cam.read()
        if frame is None:
            continue
        landmarks = extractor.detect(frame["color"], frame["timestamp_ms"])
        now = time.monotonic()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        image, _, _, _ = render(
            frame, landmarks, joint_set, cam, side, fps,
            guide_lines=guide_lines, countdown=int(np.ceil(remain)),
            footer="q 중단")
        cv2.imshow(WINDOW, image)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            return False


def show_recording(cam, extractor, joint_set, side, guide_lines):
    """'촬영 중!' 빨간 테두리 화면을 그려서 지금 찍힌다는 걸 알린다."""
    frame = cam.read()
    if frame is None:
        return
    landmarks = extractor.detect(frame["color"], frame["timestamp_ms"])
    image, _, _, _ = render(frame, landmarks, joint_set, cam, side, 0.0,
                            guide_lines=guide_lines, state="recording")
    cv2.imshow(WINDOW, image)
    cv2.waitKey(1)
