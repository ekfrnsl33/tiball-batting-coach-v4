"""
[검증] 각도 계산 방식 비교 (명세서 11번 2항: 알려진 자세로 수동 테스트)

무엇을 비교하는가:
  (1) 좌표를 만드는 방식 3가지
        2D          — 화면 좌표만 (심도 안 씀)
        P=(x,y,d)   — 명세서 2번 표기 그대로 (픽셀 + 미터 혼합)
        3D(m)       — 렌즈 정보로 세 축 모두 미터 환산
  (2) 3D를 쓸 때 '관절 위치의 거리를 어떻게 읽는가' 4가지
        중앙값 / 최소 / 하위10% / 앞면   (depth_sampling.py 참고)

  지난 검증에서 3D가 오히려 나빴는데, 원인이 심도 데이터가 아니라
  '관절 지점에서 어느 픽셀의 거리를 고르는가'였다. 손이 몸통 앞에 겹치면
  중앙값이 손과 몸통을 섞어버린다. 그래서 읽는 방식을 여러 개 만들어
  한 번의 촬영으로 전부 비교한다.

두 가지 검증:
  [A] 절대 검증 — 정답을 아는 자세(완전히 폄 180°, 직각 90°)와 비교
  [B] 방향 불변성 검증 — 팔꿈치를 고정한 채 몸만 돌린다. 참값이 안 변하므로
      방향이 바뀌어도 같은 값을 내놓는 조합이 옳다. 정답 수치를 몰라도 되고
      자세를 정확히 만들 필요도 없어서 [A]보다 신뢰할 수 있다.

원본(관절 좌표 + 심도 조각)은 test_output/raw_*.npz 에 저장한다.
다시 촬영하지 않고도 다른 방식을 시험할 수 있다.

실행 (관리자 권한 필요):
  cd ~/Desktop/tiball_project
  sudo ./venv/bin/python -u test_angle.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from angles import angle_from_record
from camera import RealSenseCamera, CameraError
from capture import (SIDE_LABEL, build_records, countdown_with_preview,
                     save_raw, show_recording, side_from_args)
from depth_sampling import SAMPLERS
from pose import PoseExtractor, LANDMARK_SETS, draw_skeleton
from textdraw import draw_text

OUTPUT_DIR = Path(__file__).parent / "test_output"
COUNTDOWN_SECONDS = 12
CAPTURE_SECONDS = 1.5
SIDE = side_from_args(sys.argv)   # --side left / --side right

JOINTS = {
    "elbow": ("shoulder", "elbow", "wrist"),
    "knee": ("hip", "knee", "ankle"),
}
JOINT_LABEL = {"elbow": "팔꿈치", "knee": "뒷무릎"}

# 비교할 조합: (표시 이름, 좌표 방식, 심도 읽는 방식)
COMBOS = [("2D (심도 미사용)", "2D", None),
          ("P=(x,y,d) 명세서", "P=(x,y,d)", "중앙값")]
COMBOS += [(f"3D + {name}", "3D(m)", name) for name in SAMPLERS]

POSES = [
    {
        "key": "side_straight",
        "short": "① 측면·다 펴기",
        "title": "① 옆으로 서서 오른 무릎·오른 팔꿈치를 곧게 펴기",
        "guide": [
            "[서는 방향] 몸을 90도 돌려, 카메라가 '오른쪽 옆면'을 보게 서세요.",
            "[자세]     차렷처럼 똑바로. 오른팔은 아래로 곧게 내려 몸 옆에.",
            "           무릎도 완전히 펴세요.  ※ 만세 아닙니다.",
        ],
        "onscreen": ["① 오른쪽 옆면이 카메라를 보게 서기",
                     "   무릎·오른팔 곧게 펴기 (차렷)"],
        "expect": {"knee": 180, "elbow": 180},
        "invariance": None,
    },
    {
        "key": "side_elbow90",
        "short": "② 측면·팔꿈치 ㄱ자",
        "title": "② 같은 방향으로 서서 오른 팔꿈치를 ㄱ자로 접기",
        "guide": [
            "[서는 방향] ①과 똑같이, 카메라가 오른쪽 옆면을 보게 그대로.",
            "[자세]     위팔(어깨~팔꿈치)은 몸통 옆에 붙여 아래로,",
            "           아래팔(팔꿈치~손목)은 바닥과 평행하게 앞으로.",
            "",
            "★ 이 팔 자세를 ③④에서도 그대로 유지하세요. 팔은 건드리지 마세요.",
        ],
        "onscreen": ["② 팔꿈치를 ㄱ자로 (아래팔은 바닥과 평행)",
                     "   ★ 이 팔 자세를 끝까지 유지하세요"],
        "expect": {"elbow": 90},
        "invariance": "측면 (0도)",
    },
    {
        "key": "turn45",
        "short": "③ 45도 회전",
        "title": "③ 팔은 그대로 두고 몸만 45도 돌기",
        "guide": [
            "[서는 방향] 그 자리에서 카메라 쪽으로 45도만 돌아서세요.",
            "[자세]     ★ 팔꿈치 각도는 절대 바꾸지 마세요. 몸만 돌립니다.",
        ],
        "onscreen": ["③ 몸만 45도 돌기 (카메라 쪽으로)",
                     "   ★ 팔꿈치 각도는 그대로!"],
        "expect": None,
        "invariance": "45도",
    },
    {
        "key": "turn90",
        "short": "④ 정면",
        "title": "④ 팔은 그대로 두고 카메라를 정면으로 마주보기",
        "guide": [
            "[서는 방향] 조금 더 돌아서 카메라를 정면으로 마주보세요.",
            "[자세]     ★ 팔꿈치 각도는 여전히 그대로.",
            "           화면에는 아래팔이 짧게 보일 겁니다. 정상입니다.",
        ],
        "onscreen": ["④ 카메라를 정면으로 마주보기",
                     "   ★ 팔꿈치 각도는 그대로!"],
        "expect": None,
        "invariance": "정면 (90도)",
    },
]


def head(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 촬영
# ---------------------------------------------------------------------------

def capture_pose(cam, extractor, pose, joint_set):
    """
    한 자세를 촬영해서 프레임별 원본 기록을 만든다.
    반환: [{x, y, v, patches, color}] (중단하면 None)
    """
    head(pose["title"])
    for line in pose["guide"]:
        print(f"  {line}")
    print(f"  {COUNTDOWN_SECONDS}초 뒤 {CAPTURE_SECONDS}초간 촬영합니다. "
          "자세를 잡고 그대로 멈춰 주세요.", flush=True)
    print("  (화면 테두리가 초록색이면 준비된 상태입니다)", flush=True)

    guide_lines = pose["onscreen"]
    if not countdown_with_preview(cam, extractor, joint_set, SIDE,
                                  COUNTDOWN_SECONDS, guide_lines):
        return None

    print("  촬영 중!", flush=True)
    show_recording(cam, extractor, joint_set, SIDE, guide_lines)
    clip = cam.record(seconds=CAPTURE_SECONDS)
    print("  촬영 끝. 분석 중...", flush=True)

    records = build_records(cam, extractor, clip)
    print(f"  분석 프레임: {len(records)}/{len(clip)}장")
    return records


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------

def angle_series(records, indices, mode, sampler, cam):
    """프레임별 각도 목록. 심도를 못 읽은 프레임은 건너뛴다."""
    values = []
    for record in records:
        angle = angle_from_record(record, indices, cam, mode, sampler)
        if angle is not None:
            values.append(angle)
    return values


def median_or_none(values):
    return float(np.median(values)) if values else None


def cell(value):
    return f"{value:8.1f}" if value is not None else "     -  "


def report(results, cam, joint_set):
    # -- [A] 절대 검증 -------------------------------------------------------
    head("[A] 절대 검증 — 정답을 아는 자세와 비교 (괄호는 오차)")

    targets = []      # (컬럼 제목, pose_key, joint, 정답)
    for pose in POSES:
        for joint, expected in (pose["expect"] or {}).items():
            if pose["key"] in results:
                targets.append((f"{pose['short'][:2]}{JOINT_LABEL[joint]}",
                                pose["key"], joint, expected))

    print(f"  {'조합':<18}" + "".join(f"{t[0]:>17}" for t in targets)
          + f"{'평균오차':>10}")
    print("  " + "-" * (18 + 17 * len(targets) + 10))

    for label, mode, sampler in COMBOS:
        cells, errors = [], []
        for _, key, joint, expected in targets:
            indices = [joint_set[role] for role in JOINTS[joint]]
            median = median_or_none(
                angle_series(results[key], indices, mode, sampler, cam))
            if median is None:
                cells.append("        -        ")
                continue
            error = abs(median - expected)
            errors.append(error)
            cells.append(f"{median:8.1f}({error:4.1f}) ")
        mean_err = f"{np.mean(errors):9.1f}°" if errors else f"{'-':>10}"
        print(f"  {label:<18}" + "".join(f"{c:>17}" for c in cells) + mean_err)

    # -- [B] 방향 불변성 검증 ------------------------------------------------
    head("[B] 방향 불변성 검증 — 팔꿈치를 고정한 채 몸만 회전")
    print("  팔꿈치 각도는 실제로 바뀌지 않았습니다.")
    print("  → 세 값이 같아야 정상. 흔들림이 곧 그 조합의 오차입니다.")
    print()

    orient = [(pose["invariance"], pose["key"]) for pose in POSES
              if pose["invariance"] and pose["key"] in results]
    indices = [joint_set[role] for role in JOINTS["elbow"]]

    print(f"  {'조합':<18}" + "".join(f"{o[0]:>13}" for o in orient)
          + f"{'흔들림':>11}{'표준편차':>11}")
    print("  " + "-" * (18 + 13 * len(orient) + 22))

    scores = {}
    for label, mode, sampler in COMBOS:
        medians = []
        cells = []
        for _, key in orient:
            median = median_or_none(
                angle_series(results[key], indices, mode, sampler, cam))
            cells.append(cell(median) + "     ")
            if median is not None:
                medians.append(median)
        if len(medians) >= 2:
            spread = max(medians) - min(medians)
            scores[label] = spread
            tail = f"{spread:10.1f}°{np.std(medians):10.1f}°"
        else:
            tail = f"{'-':>11}{'-':>11}"
        print(f"  {label:<18}" + "".join(f"{c:>13}" for c in cells) + tail)

    if scores:
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        print()
        print("  흔들림이 작은 순서:")
        for i, (label, spread) in enumerate(ranked, 1):
            print(f"    {i}. {label:<20} {spread:5.1f}°")


# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    joint_set = LANDMARK_SETS[SIDE]

    head("[준비] 카메라 연결")
    try:
        cam = RealSenseCamera().open()
    except CameraError as exc:
        print(f"[오류] {exc}")
        sys.exit(1)

    results = {}
    try:
        print("  머리부터 발목까지 화면에 들어오는 위치에서 진행해주세요.")
        print(f"  자세 {len(POSES)}가지를 차례로 촬영합니다. (오른팔·오른다리 기준)")
        print()
        print("  용어: '위팔' = 어깨~팔꿈치 / '아래팔' = 팔꿈치~손목")
        print("  ②에서 잡은 팔꿈치 각도를 ③④까지 유지하는 것이 핵심입니다.")

        with PoseExtractor() as extractor:
            for pose in POSES:
                records = capture_pose(cam, extractor, pose, joint_set)
                if records is None:
                    print("\n[중단] 사용자가 중단했습니다.")
                    break
                if not records:
                    print("  [주의] 사람을 찾은 프레임이 없어 건너뜁니다.")
                    continue
                results[pose["key"]] = records
                save_raw(OUTPUT_DIR / f"raw_{pose['key']}.npz", records, cam)
                save_check_image(pose, records, cam, joint_set)
    finally:
        cv2.destroyAllWindows()
        cam.close()

    if results:
        report(results, cam, joint_set)


def save_check_image(pose, records, cam, joint_set):
    """중앙값 프레임에 각 조합의 각도를 적어서 저장한다."""
    joint = next(iter(pose["expect"] or {"elbow": None}))
    indices = [joint_set[role] for role in JOINTS[joint]]
    series = angle_series(records, indices, "3D(m)", "앞면", cam)
    if not series:
        series = angle_series(records, indices, "2D", None, cam)
    if not series:
        return
    median = float(np.median(series))

    # 중앙값에 가장 가까운 프레임 찾기
    def frame_angle(rec):
        vals = angle_series([rec], indices, "3D(m)", "앞면", cam) \
            or angle_series([rec], indices, "2D", None, cam)
        return vals[0] if vals else 1e9

    best = min(records, key=lambda r: abs(frame_angle(r) - median))
    overlay = draw_skeleton(best["color"], best["landmarks"],
                            highlight_indices=sorted(set(joint_set.values())))
    items = [(f"{pose['short']}  ({JOINT_LABEL[joint]})", (12, 8), 21,
              (245, 245, 245))]
    for i, (label, mode, sampler) in enumerate(COMBOS):
        vals = angle_series([best], indices, mode, sampler, cam)
        text = f"{label}: {vals[0]:.1f}도" if vals else f"{label}: 측정불가"
        items.append((text, (12, 34 + i * 24), 19, (40, 200, 240)))
    path = OUTPUT_DIR / f"angle_{pose['key']}.png"
    cv2.imwrite(str(path), draw_text(overlay, items))
    print(f"  확인용 이미지: {path.name} (중앙값 프레임)")


if __name__ == "__main__":
    main()
