"""
[2단계] RealSense + MediaPipe 연결 테스트 (명세서 11번 1항)

목적:
  각도 계산·판정은 아직 넣지 않고, "관절이 제대로 잡히는지"만 확인한다.

확인하는 것:
  1) 2초 녹화가 되는지 (명세서 2번: 실시간 스트리밍 아닌 단기 녹화)
  2) 프레임마다 사람의 관절 33개가 잡히는지
  3) 판정에 쓰는 6개 지점의 가시성(visibility)이 임계값 γ를 넘는지
  4) 각 관절 위치에서 RealSense 실측 심도가 읽히는지 (명세서 2번의 P=(x,y,d))
  5) 골격선 오버레이 이미지가 사람 몸에 잘 맞는지 (눈으로 확인)

실행 (관리자 권한 필요):
  cd ~/Desktop/tiball_project
  sudo ./venv/bin/python test_pose.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from camera import RealSenseCamera, CameraError
from pose import (PoseExtractor, LANDMARK_SETS, LANDMARK_NAMES,
                  VISIBILITY_THRESHOLD, draw_skeleton, is_visible)

# 명세서 2번: 2~4초 단기 녹화. 아이가 스윙 타이밍을 맞출 여유를 주려면
# 상한인 4초가 안전하다.
RECORD_SECONDS = 4.0
COUNTDOWN_SECONDS = 5
OUTPUT_DIR = Path(__file__).parent / "test_output"

# 이 테스트에서는 좌우 양쪽 다 확인한다.
# (실제 앱에서는 명세서 3번대로 연구자가 좌타/우타를 미리 선택한다)
BATTING_SIDE = "right"


def head(title):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    head(f"[1] 카메라 연결 + {RECORD_SECONDS:g}초 녹화")
    try:
        cam = RealSenseCamera().open()
    except CameraError as exc:
        print(f"[오류] {exc}")
        sys.exit(1)

    try:
        print(f"  심도 스케일: {cam.depth_scale}")
        print()
        print("  카메라 앞에 머리부터 발목까지 다 들어오게 서 주세요.")
        print("  (뒷무릎 각도는 골반-무릎-발목 세 점으로 재므로 발목이 꼭 필요합니다)")
        print(f"  {COUNTDOWN_SECONDS}초 뒤 {RECORD_SECONDS}초간 녹화합니다.")
        # 명세서 8번 ②촬영 진행 화면의 카운트다운에 해당하는 부분.
        # 프레임 개수로 시간을 세면 안 된다 — 카메라에 프레임이 미리 쌓여
        # 있으면 순식간에 읽혀버려서 카운트다운이 훅 지나간다.
        # 실제 시계로 재고, 그 사이 프레임은 계속 버려서 최신 상태를 유지한다.
        for remain in range(COUNTDOWN_SECONDS, 0, -1):
            print(f"    {remain}...", flush=True)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                cam.read()
        print("  촬영 중!", flush=True)
        clip = cam.record(seconds=RECORD_SECONDS)
        print("  촬영 끝.", flush=True)
        if not clip:
            print("[오류] 녹화된 프레임이 없습니다.")
            sys.exit(1)

        # 60fps는 노출이 짧아져 어두워질 수 있으니 밝기를 확인한다.
        # 경험적으로 평균 밝기가 40 아래면 너무 어둡다고 본다.
        brightness = float(np.mean([f["color"].mean() for f in clip]))
        depth_valid = float(np.mean([(f["depth"] > 0).mean() for f in clip])) * 100
        print(f"  평균 밝기: {brightness:.1f} / 255"
              f"{'  ⚠ 너무 어둡습니다 (조명을 켜거나 fps를 30으로 낮추세요)' if brightness < 40 else '  (적정)'}")
        print(f"  심도 유효 픽셀: {depth_valid:.1f}%")

        head("[2] MediaPipe로 관절 추출")
        joint_set = LANDMARK_SETS[BATTING_SIDE]
        tracked = sorted(set(joint_set.values())
                         | set(LANDMARK_SETS["left"].values()))

        detected_count = 0
        # 지점별로 가시성/심도가 잘 나오는지 누적 집계
        vis_records = {idx: [] for idx in tracked}
        depth_ok = {idx: 0 for idx in tracked}
        in_frame_ok = {idx: 0 for idx in tracked}
        per_frame = []

        h, w = clip[0]["color"].shape[:2]

        with PoseExtractor() as extractor:
            for i, frame in enumerate(clip):
                landmarks = extractor.detect(frame["color"],
                                             frame["timestamp_ms"])
                per_frame.append(landmarks)
                if landmarks is None:
                    continue
                detected_count += 1
                for idx in tracked:
                    lm = landmarks[idx]
                    vis_records[idx].append(lm["visibility"])
                    # MediaPipe는 화면 밖 위치도 추정해서 내놓는다.
                    # 화면 밖이면 심도를 읽을 수가 없으니 따로 세어둔다.
                    if 0 <= lm["x"] < w and 0 <= lm["y"] < h:
                        in_frame_ok[idx] += 1
                    if cam.sample_depth(frame["depth"], lm["x"], lm["y"]) is not None:
                        depth_ok[idx] += 1

        total = len(clip)
        print(f"  녹화 프레임: {total}장")
        print(f"  사람 감지 성공: {detected_count}장 "
              f"({detected_count / total * 100:.0f}%)")

        if detected_count == 0:
            print()
            print("[실패] 어떤 프레임에서도 사람을 찾지 못했습니다.")
            print("       카메라 앞에 전신(적어도 무릎까지)이 들어오게 서서 다시 실행해주세요.")
            sys.exit(2)

        head(f"[3] 판정에 쓰는 지점별 상태 (γ = {VISIBILITY_THRESHOLD})")
        print(f"  {'지점':<10} {'평균 가시성':>10} {'γ 통과':>9} "
              f"{'화면 안':>9} {'심도 측정':>9}")
        print("  " + "-" * 54)
        out_of_frame = []
        for idx in tracked:
            vis = np.array(vis_records[idx]) if vis_records[idx] else np.array([0.0])
            pass_ratio = float((vis >= VISIBILITY_THRESHOLD).mean()) * 100
            frame_ratio = in_frame_ok[idx] / detected_count * 100
            depth_ratio = depth_ok[idx] / detected_count * 100
            ok = pass_ratio > 90 and depth_ratio > 90
            print(f"  {LANDMARK_NAMES[idx]:<10} {vis.mean():>10.2f} "
                  f"{pass_ratio:>8.0f}% {frame_ratio:>8.0f}% "
                  f"{depth_ratio:>8.0f}%{'' if ok else ' ⚠'}")
            if frame_ratio < 50:
                out_of_frame.append(LANDMARK_NAMES[idx])

        if out_of_frame:
            print()
            print(f"  ⚠ 화면 밖으로 나간 지점: {', '.join(out_of_frame)}")
            # 얼마나 모자란지 픽셀로 알려주면 카메라를 얼마나 옮겨야 할지 감이 온다.
            # (MediaPipe는 화면 밖 위치도 추정해서 내놓지만 심도는 읽을 수 없다)
            below = max((lm["y"] for lms in per_frame if lms
                         for idx in tracked for lm in [lms[idx]]), default=0)
            headroom = min((lm["y"] for lms in per_frame if lms
                            for idx in tracked for lm in [lms[idx]]), default=0)
            if below >= h:
                print(f"    아래로 약 {below - h + 1:.0f}픽셀 부족 "
                      f"(머리 위 여유는 {headroom:.0f}픽셀)")
                if headroom > below - h:
                    print("    → 카메라를 살짝 아래로 기울이면 됩니다 (뒤로 갈 필요 없음)")
                else:
                    print("    → 카메라에서 30~50cm 더 뒤로 떨어져 주세요")

        head("[4] 실측 심도 결합 확인 (명세서 2번: P = (x, y, d))")
        # 가운데 프레임 하나를 골라 6개 지점의 좌표와 거리를 보여준다
        mid = next((i for i in range(len(clip) // 2, len(clip))
                    if per_frame[i] is not None), None)
        if mid is None:
            mid = next(i for i, lms in enumerate(per_frame) if lms is not None)

        frame = clip[mid]
        landmarks = per_frame[mid]
        print(f"  {mid + 1}번째 프레임 기준 ({BATTING_SIDE} = "
              f"{'우타' if BATTING_SIDE == 'right' else '좌타'}자 세트)")
        print(f"  {'지점':<10} {'화면 x':>8} {'화면 y':>8} {'거리(m)':>9}"
              f"  {'실제 3D 좌표 (m)':<28} 가시성")
        print("  " + "-" * 78)
        for role, idx in joint_set.items():
            lm = landmarks[idx]
            d = cam.sample_depth(frame["depth"], lm["x"], lm["y"])
            if d is None:
                coord = "(심도 측정 실패)"
                dist = "  -  "
            else:
                x3, y3, z3 = cam.pixel_to_point(lm["x"], lm["y"], d)
                coord = f"({x3:+.3f}, {y3:+.3f}, {z3:+.3f})"
                dist = f"{d:.3f}"
            flag = "" if is_visible(lm) else "  ← γ 미달"
            print(f"  {LANDMARK_NAMES[idx]:<10} {lm['x']:>8.1f} {lm['y']:>8.1f} "
                  f"{dist:>9}  {coord:<28} {lm['visibility']:.2f}{flag}")

        head("[5] 골격선 오버레이 이미지 저장")
        saved = []
        picks = [("first", next(i for i, l in enumerate(per_frame) if l is not None)),
                 ("mid", mid),
                 ("last", max(i for i, l in enumerate(per_frame) if l is not None))]
        for label, i in picks:
            overlay = draw_skeleton(clip[i]["color"], per_frame[i],
                                    highlight_indices=tracked)
            path = OUTPUT_DIR / f"pose_{label}.png"
            cv2.imwrite(str(path), overlay)
            saved.append(path)
            print(f"   {path}  ({i + 1}번째 프레임)")

        print()
        print("  주황색 큰 점 = 판정에 쓰는 12개 지점(좌/우), 초록 굵은 선 = 그 연결")
        print("  회색 작은 점 = 나머지 랜드마크")

        print()
        if detected_count == total:
            print("[성공] 모든 프레임에서 관절이 잡혔습니다. 이미지로 위치가 맞는지 확인해주세요.")
        else:
            print(f"[주의] {total}장 중 {detected_count}장에서만 잡혔습니다. "
                  "전신이 화면에 들어오는지, 조명이 충분한지 확인해주세요.")

    finally:
        cam.close()


if __name__ == "__main__":
    main()
