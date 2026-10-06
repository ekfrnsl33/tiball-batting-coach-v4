"""
[재분석] 저장된 원본(raw_*.npz)으로 다시 계산한다 — 카메라 없이, 촬영 없이

쓰는 경우:
  - 좌타/우타를 잘못 지정해서 촬영했을 때 (관절 33개를 다 저장해두므로
    반대쪽으로 다시 계산하면 된다)
  - 심도 읽는 방법이나 좌표 방식을 바꿔서 비교해볼 때

실행 (관리자 권한 필요 없음):
  cd ~/Desktop/tiball_project
  ./venv/bin/python reanalyze.py test_output/raw_impact.npz --side left
"""

import sys
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

import impact as impact_mod
from angles import DEFAULT_MODE, angle_from_record
from capture import SIDE_LABEL, side_from_args
from depth_sampling import SAMPLERS
from pose import (LANDMARK_NAMES, LANDMARK_SETS, VISIBILITY_THRESHOLD)

FPS = 60      # 촬영에 쓴 프레임레이트

JOINTS = {
    "elbow": ("shoulder", "elbow", "wrist"),
    "knee": ("hip", "knee", "ankle"),
}
JOINT_LABEL = {"elbow": "팔꿈치", "knee": "뒷무릎"}
REFERENCE = {"elbow": 100.5, "knee": 109.6}      # 명세서 5번


class OfflineCamera:
    """
    저장된 파일로 카메라 흉내를 낸다.
    camera.RealSenseCamera에서 계산에 필요한 부분(심도 스케일, 렌즈 정보)만 갖춘다.
    """

    def __init__(self, intrinsics, depth_scale, width=640, height=480):
        fx, fy, ppx, ppy = [float(v) for v in intrinsics]
        self.depth_scale = float(depth_scale)
        self.intrinsics = rs.intrinsics()
        self.intrinsics.width = width
        self.intrinsics.height = height
        self.intrinsics.fx, self.intrinsics.fy = fx, fy
        self.intrinsics.ppx, self.intrinsics.ppy = ppx, ppy
        self.intrinsics.model = rs.distortion.brown_conrady
        self.intrinsics.coeffs = [0.0] * 5
        self.fps = FPS

    def pixel_to_point(self, x, y, depth_m):
        return rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [float(x), float(y)], float(depth_m))


def load(path):
    data = np.load(path)
    cam = OfflineCamera(data["intrinsics"], data["depth_scale"])
    records = [{"x": data["x"][i], "y": data["y"][i],
                "v": data["visibility"][i], "patches": data["patches"][i]}
               for i in range(len(data["x"]))]
    return records, cam


def head(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print("사용법: ./venv/bin/python reanalyze.py <raw_*.npz> "
              "[--side left|right]")
        sys.exit(1)
    path = Path(args[0])
    if not path.exists():
        print(f"[오류] 파일이 없습니다: {path}")
        sys.exit(1)

    side = side_from_args(sys.argv)
    records, cam = load(path)
    joint_set = LANDMARK_SETS[side]

    head(f"[재분석] {path.name} — {SIDE_LABEL[side]}")
    print(f"  프레임 {len(records)}장, {FPS}fps")
    print(f"  측정 관절: "
          f"{', '.join(LANDMARK_NAMES[i] for i in joint_set.values())}")

    # -- 임팩트 프레임 -------------------------------------------------------
    head("[1] 임팩트 프레임 (명세서 6번: 손목 최고 속도)")
    wrist = [LANDMARK_SETS["left"]["wrist"],
             LANDMARK_SETS["right"]["wrist"]]
    results = {}
    for label, mode in (("3차원 (m/s)", "3D(m)"), ("화면 (픽셀/s)", "2D")):
        result = impact_mod.find_impact(records, wrist, cam, FPS, mode)
        if result is None:
            continue
        results[label] = result
        unit = "m/s" if mode == "3D(m)" else "px/s"
        print(f"  {label:<14} {result['index']:3d}번 "
              f"({result['index'] / FPS:.2f}초), "
              f"최고 {result['peak_speed']:8.2f} {unit}")
        for index, speed in result["competing"]:
            print(f"      ⚠ 경쟁 봉우리 {index:3d}번 "
                  f"({speed / result['peak_speed'] * 100:.0f}%)")

    primary = results.get("3차원 (m/s)") or next(iter(results.values()))
    peak = primary["index"]

    # -- 임팩트 프레임의 각도 (심도 읽는 방법별) -----------------------------
    head(f"[2] 임팩트({peak}번) 각도 — 심도 읽는 방법별 비교")
    record = records[peak]
    print(f"  {'방법':<12}" + "".join(f"{JOINT_LABEL[j]:>12}" for j in JOINTS)
          + f"{'기준각 차이':>24}")
    print("  " + "-" * 62)

    rows = [("2D (심도 미사용)", "2D", None)]
    rows += [(f"3D + {name}", "3D(m)", name) for name in SAMPLERS]
    for label, mode, sampler in rows:
        cells, diffs = [], []
        for joint in JOINTS:
            indices = [joint_set[role] for role in JOINTS[joint]]
            angle = angle_from_record(record, indices, cam, mode, sampler)
            if angle is None:
                cells.append(f"{'측정불가':>12}")
                diffs.append("     -  ")
            else:
                cells.append(f"{angle:11.1f}°")
                diffs.append(f"{angle - REFERENCE[joint]:+7.1f}°")
        print(f"  {label:<12}" + "".join(cells)
              + "   " + "  ".join(diffs))

    print()
    print(f"  기준각: 팔꿈치 {REFERENCE['elbow']}° / 뒷무릎 {REFERENCE['knee']}°"
          "  (명세서 5번)")

    # -- 가시성 -------------------------------------------------------------
    head("[3] 임팩트 프레임의 가시성 (명세서 3번 γ 필터)")
    print(f"  γ = {VISIBILITY_THRESHOLD}")
    for joint in JOINTS:
        indices = [joint_set[role] for role in JOINTS[joint]]
        vis = [float(record["v"][i]) for i in indices]
        passed = all(v >= VISIBILITY_THRESHOLD for v in vis)
        detail = "  ".join(f"{LANDMARK_NAMES[i]} {v:.2f}"
                           for i, v in zip(indices, vis))
        print(f"  {JOINT_LABEL[joint]:<7} {detail}"
              f"   → {'통과' if passed else 'γ 미달'}")

    # -- 스윙 전체 각도 범위 -------------------------------------------------
    head("[4] 스윙 전체에서 각도가 지나간 범위")
    for joint in JOINTS:
        indices = [joint_set[role] for role in JOINTS[joint]]
        values = [angle_from_record(r, indices, cam, DEFAULT_MODE)
                  for r in records]
        values = np.array([v for v in values if v is not None])
        if values.size == 0:
            continue
        print(f"  {JOINT_LABEL[joint]:<7} 최소 {values.min():6.1f}°  "
              f"최대 {values.max():6.1f}°  "
              f"중앙값 {np.median(values):6.1f}°  "
              f"(기준각 {REFERENCE[joint]}° 부근 프레임 "
              f"{np.mean(np.abs(values - REFERENCE[joint]) <= 10) * 100:.0f}%)")


if __name__ == "__main__":
    main()
