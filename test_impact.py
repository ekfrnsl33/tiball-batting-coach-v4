"""
[3단계] 임팩트 프레임 탐지 (명세서 6번) + 그 프레임의 각도 산출

무엇을 확인하는가:
  1) 스윙 4초를 녹화해서, 손목이 가장 빠른 프레임을 임팩트로 찾아내는지
  2) 그 프레임이 정말 타격 순간인지 (앞뒤 프레임 그림으로 눈으로 확인)
  3) 그 프레임에서 팔꿈치·뒷무릎 각도가 나오는지 (명세서 4번 공식, 3D 좌표)
  4) 픽셀 속도와 3차원 속도 중 어느 쪽이 봉우리를 더 잘 찾는지

만들어지는 것 (test_output/):
  impact_speed.png    속도 곡선 + 임팩트 표시
  impact_frames.png   임팩트 앞뒤 프레임 골격선 + 각도
  raw_impact.npz      원본 (재분석용)

실행 (관리자 권한 필요):
  cd ~/Desktop/tiball_project
  sudo ./venv/bin/python -u test_impact.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

import impact as impact_mod
from angles import (ANGLE_WINDOW_HALF, DEFAULT_MODE, angle_around,
                    angle_from_record)
from camera import RealSenseCamera, CameraError
from capture import (SIDE_LABEL, build_records, countdown_with_preview,
                     save_raw, show_recording, side_from_args)
from chart import draw_series
from pose import (LANDMARK_SETS, LANDMARK_NAMES, PoseExtractor,
                  VISIBILITY_THRESHOLD, draw_skeleton)
from textdraw import draw_text

OUTPUT_DIR = Path(__file__).parent / "test_output"

# 명세서 2번은 2~4초. 4초로 했더니 스윙 뒤 다른 동작(배트 다시 들기 등)이
# 섞여서 그쪽이 더 빠르게 잡히는 일이 있었다. 3초로 줄이고 '스윙 후 정지'를
# 안내한다.
RECORD_SECONDS = 3.0
COUNTDOWN_SECONDS = 10

# 명세서 3번: 좌타/우타는 자동 인식 없이 사전 선택.
# 실행할 때 --side left / --side right 로 고른다 (기본 좌타).
SIDE = side_from_args(sys.argv)

# 명세서 5번 기준각 (판정은 4단계에서 하지만, 참고로 함께 보여준다)
THETA_E = 100.5
THETA_K = 109.6

JOINTS = {
    "elbow": ("shoulder", "elbow", "wrist"),
    "knee": ("hip", "knee", "ankle"),
}
JOINT_LABEL = {"elbow": "팔꿈치", "knee": "뒷무릎"}
REFERENCE = {"elbow": THETA_E, "knee": THETA_K}

GUIDE = [
    "빨간 테두리가 되면 바로 스윙 한 번",
    "스윙 뒤에는 그대로 멈춰 계세요 (다른 동작이 섞이면 오탐)",
]


def head(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def joint_indices(joint_set, joint):
    return [joint_set[role] for role in JOINTS[joint]]


def save_frame_strip(records, peak, joint_set, cam):
    """임팩트 앞뒤 프레임을 나란히 붙여서 눈으로 확인할 수 있게 저장한다."""
    offsets = [-4, -2, 0, 2, 4]
    tiles = []
    for offset in offsets:
        i = peak + offset
        if not (0 <= i < len(records)):
            continue
        record = records[i]
        overlay = draw_skeleton(
            record["color"], record["landmarks"],
            highlight_indices=sorted(set(joint_set.values())))
        tile = cv2.resize(overlay, (400, 300))
        label = "임팩트" if offset == 0 else f"{offset:+d}"
        items = [(f"{label}  ({i}번)", (10, 6), 22,
                  (60, 60, 235) if offset == 0 else (235, 235, 235))]
        for row, joint in enumerate(JOINTS):
            angle = angle_from_record(record, joint_indices(joint_set, joint),
                                      cam)
            text = (f"{JOINT_LABEL[joint]} {angle:.1f}도" if angle is not None
                    else f"{JOINT_LABEL[joint]} 측정불가")
            items.append((text, (10, 32 + row * 24), 19, (40, 200, 240)))
        if offset == 0:
            cv2.rectangle(tile, (0, 0), (399, 299), (60, 60, 235), 4)
        tiles.append(draw_text(tile, items))

    if tiles:
        path = OUTPUT_DIR / "impact_frames.png"
        cv2.imwrite(str(path), np.hstack(tiles))
        print(f"  임팩트 앞뒤 프레임: {path.name}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    joint_set = LANDMARK_SETS[SIDE]

    head(f"[준비] {SIDE_LABEL[SIDE]} 기준으로 측정합니다")
    print("  다른 쪽이면 --side left / --side right 로 다시 실행하세요.")
    print(f"  측정 관절: {', '.join(LANDMARK_NAMES[i] for i in joint_set.values())}")

    try:
        cam = RealSenseCamera().open()
    except CameraError as exc:
        print(f"[오류] {exc}")
        sys.exit(1)

    try:
        print("  머리부터 발목까지 화면에 들어오는 위치에서 진행해주세요.")
        print(f"  {COUNTDOWN_SECONDS}초 뒤 {RECORD_SECONDS:g}초간 녹화합니다.")
        print("  녹화가 시작되면(빨간 테두리) 바로 스윙 한 번 하고,", flush=True)
        print("  스윙이 끝나면 그대로 멈춰 계세요.", flush=True)

        with PoseExtractor() as extractor:
            if not countdown_with_preview(cam, extractor, joint_set, SIDE,
                                          COUNTDOWN_SECONDS, GUIDE):
                print("\n[중단] 사용자가 중단했습니다.")
                return
            print("  촬영 중! 스윙하세요!", flush=True)
            show_recording(cam, extractor, joint_set, SIDE, GUIDE)
            clip = cam.record(seconds=RECORD_SECONDS)
            print("  촬영 끝. 분석 중...", flush=True)
            records = build_records(cam, extractor, clip)
    finally:
        cv2.destroyAllWindows()
        cam.close()

    if len(records) < 10:
        print(f"[오류] 사람이 잡힌 프레임이 {len(records)}장뿐입니다. 다시 촬영해주세요.")
        sys.exit(2)

    save_raw(OUTPUT_DIR / "raw_impact.npz", records, cam)
    fps = cam.fps
    print(f"  분석 프레임: {len(records)}/{len(clip)}장 ({fps}fps)")

    # -----------------------------------------------------------------------
    # 임팩트 찾기 — 픽셀 속도와 3차원 속도를 각각
    # -----------------------------------------------------------------------
    head("[1] 임팩트 프레임 탐지 (명세서 6번: 손목 최고 속도)")
    # 양손을 함께 쓴다 (배트는 두 손으로 잡으므로 임팩트에서 둘 다 빨라진다).
    # 한쪽만 쓰면 그 손이 몸에 가려 잘못 추적될 때 엉뚱한 프레임이 뽑힌다.
    wrist = [LANDMARK_SETS["left"]["wrist"], LANDMARK_SETS["right"]["wrist"]]
    print(f"  기준 관절: {', '.join(LANDMARK_NAMES[i] for i in wrist)} "
          "(둘 중 빠른 쪽)")

    found = {}
    for label, mode in (("3차원 (m/s)", "3D(m)"), ("화면 (픽셀/s)", "2D")):
        result = impact_mod.find_impact(records, wrist, cam, fps, mode)
        if result is None:
            print(f"  {label:<14} 계산 불가")
            continue
        found[label] = result
        unit = "m/s" if mode == "3D(m)" else "px/s"
        warn = "  ⚠ 녹화 끝에 걸려 스윙이 잘렸을 수 있음" if result["near_edge"] else ""
        print(f"  {label:<14} {result['index']:3d}번 프레임 "
              f"({result['index'] / fps:.2f}초), 최고속도 "
              f"{result['peak_speed']:7.2f} {unit}, "
              f"계산 가능 프레임 {result['valid_ratio'] * 100:.0f}%{warn}")

    if not found:
        print("[오류] 속도를 계산할 수 없었습니다.")
        sys.exit(2)

    indices = [r["index"] for r in found.values()]
    if len(indices) == 2:
        gap = abs(indices[0] - indices[1])
        print()
        print(f"  두 방식이 고른 프레임 차이: {gap}프레임 "
              f"({gap / fps * 1000:.0f}ms)")
        if gap <= 2:
            print("  → 거의 같은 지점을 골랐습니다. 어느 쪽을 써도 무방합니다.")
        else:
            print("  → 차이가 큽니다. 아래 그림으로 어느 쪽이 타격 순간인지 확인해주세요.")

    # 스윙 말고 다른 큰 움직임이 섞였는지 확인.
    # 명세서 6번은 '전체 구간의 최댓값'을 쓰므로, 비슷한 크기의 봉우리가
    # 또 있으면 그 움직임이 조금만 더 빨랐어도 엉뚱한 프레임이 뽑힌다.
    for label, result in found.items():
        if result["competing"]:
            print()
            print(f"  ⚠ [{label}] 최고 봉우리({result['index']}번) 말고도 "
                  f"비슷한 크기의 움직임이 있습니다:")
            for index, speed in result["competing"]:
                print(f"      {index:3d}번 프레임 ({index / fps:.2f}초) "
                      f"속도 {speed:.2f} "
                      f"— 최고의 {speed / result['peak_speed'] * 100:.0f}%")
            print("    → 스윙 외의 동작(배트 다시 들기, 걸어 나오기 등)이 섞인 것으로 보입니다.")

    # 각도는 3차원으로 재기로 했으므로(2026-09-05 검증) 3차원 속도 기준을 우선
    primary = found.get("3차원 (m/s)") or next(iter(found.values()))
    peak = primary["index"]

    # -----------------------------------------------------------------------
    # 임팩트 프레임의 각도
    # -----------------------------------------------------------------------
    head(f"[2] 임팩트 프레임({peak}번)의 각도 — 명세서 4번 공식, 3D 좌표")
    record = records[peak]
    print(f"  {'관절':<7} {'그 프레임':>9} {'주변 중앙값':>11} {'흔들림':>8} "
          f"{'기준각':>7} {'차이':>8}   가시성")
    print("  " + "-" * 74)
    for joint in JOINTS:
        idx = joint_indices(joint_set, joint)
        single = angle_from_record(record, idx, cam, DEFAULT_MODE)
        # 한 프레임만 쓰면 하필 튄 프레임을 집을 수 있어 주변 중앙값도 함께 본다
        median, spread, used = angle_around(records, idx, cam, peak)
        vis = [float(record["v"][i]) for i in idx]
        passed = all(v >= VISIBILITY_THRESHOLD for v in vis)
        vis_text = ("  ".join(f"{v:.2f}" for v in vis)
                    + ("  통과" if passed else "  ← γ 미달"))
        one = f"{single:8.1f}°" if single is not None else f"{'측정불가':>9}"
        if median is None:
            print(f"  {JOINT_LABEL[joint]:<7} {one} {'-':>11} {'-':>8} "
                  f"{REFERENCE[joint]:7.1f} {'-':>8}   {vis_text}")
        else:
            print(f"  {JOINT_LABEL[joint]:<7} {one} {median:10.1f}° "
                  f"{spread:7.1f}° {REFERENCE[joint]:7.1f} "
                  f"{median - REFERENCE[joint]:+7.1f}°   {vis_text}")
    print()
    print(f"  '주변 중앙값'은 임팩트 ±{ANGLE_WINDOW_HALF}프레임의 중앙값입니다.")
    print("  '흔들림'이 크면 그 구간에서 관절 추적이 불안정하다는 뜻입니다.")
    print()
    print("  ※ 판정(완벽/좋음/다시연습)은 4단계에서 붙입니다. 여기선 수치만 봅니다.")

    # -----------------------------------------------------------------------
    # 스윙 구간 전체의 각도 변화 + 속도 곡선
    # -----------------------------------------------------------------------
    head("[3] 확인용 이미지 저장")
    series = {}
    for label, result in found.items():
        series[label] = result["smoothed"]
    chart = draw_series(series, peak_index=peak, title="손목 속도 (다듬은 값)",
                        x_label="프레임")
    path = OUTPUT_DIR / "impact_speed.png"
    cv2.imwrite(str(path), chart)
    print(f"  속도 곡선: {path.name}")

    angle_series = {}
    for joint in JOINTS:
        idx = joint_indices(joint_set, joint)
        angle_series[JOINT_LABEL[joint]] = [
            angle_from_record(r, idx, cam, DEFAULT_MODE) or np.nan
            for r in records]
    chart2 = draw_series(angle_series, peak_index=peak,
                         title="관절 각도 변화 (도)", x_label="프레임")
    path2 = OUTPUT_DIR / "impact_angles.png"
    cv2.imwrite(str(path2), chart2)
    print(f"  각도 곡선: {path2.name}")

    save_frame_strip(records, peak, joint_set, cam)

    print()
    print("  impact_frames.png 에서 가운데(빨간 테두리)가 정말 타격 순간인지")
    print("  확인해주세요. 어긋나 있으면 탐지 방식을 손봐야 합니다.")


if __name__ == "__main__":
    main()
