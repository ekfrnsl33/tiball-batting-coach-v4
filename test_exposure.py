"""
[추가 검증] 노출 시간이 관절 추적 정확도에 미치는 영향

배경 (2026-09-05 1차 실측):
  임팩트 구간에서 팔꿈치 각도가 크게 튀었다. 원인은 스윙이 가장 빠른
  구간의 모션 블러로 보인다. 노출 3가지를 한 번씩 찍어 비교했더니
  고정 4ms가 가장 좋았지만(팔꿈치 불확실성 ±1.7°), 그 스윙이 가장
  느렸기 때문에(307°/초 vs 487·620°/초) 노출 효과와 스윙 속도가 섞여
  있었다. 그래서 같은 설정을 여러 번 찍어 확인한다.

측정 방법 — '직선 맞춤':
  임팩트 순간 팔꿈치는 초당 300~600°로 실제로 빠르게 펴지는 중이다.
  그래서 주변 프레임의 표준편차를 '흔들림'으로 쓰면 실제 변화까지
  잡음으로 세어버린다 (1차 때 제가 한 실수).
  주변 창에 직선을 맞추면 실제 추세는 직선이 흡수하고, 직선에서 벗어난
  정도만 잡음이 된다. 그 직선을 임팩트 프레임에서 읽은 값이 추정 각도이고,
  잡음이 프레임 수만큼 평균되어 불확실성이 줄어든다.

판단 기준:
  명세서 5번 허용오차는 팔꿈치 α=3°/β=6°, 뒷무릎 α=5°/β=10°.
  불확실성이 α보다 작아야 3단계(배율 ×1) 판정이 성립한다.

실행 (관리자 권한 필요):
  cd ~/Desktop/tiball_project
  sudo ./venv/bin/python -u test_exposure.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

import impact as impact_mod
from angles import angle_fit_at, angle_from_record
from camera import RealSenseCamera, CameraError
from capture import (SIDE_LABEL, build_records, countdown_with_preview,
                     save_raw, show_recording, side_from_args)
from pose import LANDMARK_SETS, PoseExtractor

OUTPUT_DIR = Path(__file__).parent / "test_output"

RECORD_SECONDS = 3.0
COUNTDOWN_SECONDS = 10
SIDE = side_from_args(sys.argv)

JOINTS = {
    "elbow": ("shoulder", "elbow", "wrist"),
    "knee": ("hip", "knee", "ankle"),
}
JOINT_LABEL = {"elbow": "팔꿈치", "knee": "뒷무릎"}
TOLERANCE = {"elbow": 3, "knee": 5}          # 명세서 5번 α

# (이름, 방식, 값)
#   auto        : 지금까지 쓴 설정 (거실에서 16.6ms를 꽉 채워 씀 → 번짐)
#   auto_limit  : 자동노출을 켜둔 채 노출 시간의 상한만 제한
#
# 고정 노출이 아니라 '상한'을 쓰는 이유: 고정하면 조명이 다른 장소
# (예: 학교 체육관)에서 너무 밝거나 어두워져 다시 맞춰야 한다.
# 상한만 걸면 밝기는 카메라가 알아서 맞추고 번짐만 억제된다.
SETTINGS = [
    ("자동노출", "auto", None),
    ("자동+4ms상한", "auto_limit", 4.0),
]
REPEATS = 3


def head(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def joint_indices(joint_set, joint):
    return [joint_set[role] for role in JOINTS[joint]]


def measure(records, cam, joint_set):
    """한 번의 촬영에서 임팩트 각도와 그 불확실성을 뽑는다."""
    wrists = [LANDMARK_SETS["left"]["wrist"], LANDMARK_SETS["right"]["wrist"]]
    result = impact_mod.find_impact(records, wrists, cam, cam.fps, "3D(m)")
    if result is None:
        return None
    peak = result["index"]

    out = {
        "brightness": float(np.mean([r["color"].mean() for r in records])),
        "impact": peak,
        "peak_speed": result["peak_speed"],
        "competing": len(result["competing"]),
        "joints": {},
    }
    for joint in JOINTS:
        indices = joint_indices(joint_set, joint)
        fit = angle_fit_at(records, indices, cam, peak)
        out["joints"][joint] = {
            "single": angle_from_record(records[peak], indices, cam),
            "angle": fit["angle"] if fit else None,
            "noise": fit["noise"] if fit else None,
            "uncertainty": fit["uncertainty"] if fit else None,
            "rate": (abs(fit["rate"]) * cam.fps
                     if fit and fit.get("rate") is not None else None),
        }
    return out


def report(results):
    head("[비교] 노출 설정별 관절 추적 정확도")
    print("  '불확실성' = 임팩트 각도 추정값을 얼마나 믿을 수 있는지")
    print("  명세서 5번 허용오차 α: 팔꿈치 3° / 뒷무릎 5°")
    print("  → 불확실성이 α보다 작아야 3단계(배율 ×1) 판정이 성립합니다.")

    for joint in JOINTS:
        print()
        print(f"  [{JOINT_LABEL[joint]}]  (α = {TOLERANCE[joint]}°)")
        print(f"    {'설정':<13}{'회차':>4}{'노출':>8}{'밝기':>6}{'각도':>9}"
              f"{'불확실성':>10}{'잡음':>8}{'변화속도':>11}")
        print("    " + "-" * 70)
        for name, takes in results.items():
            values = []
            for i, stats in enumerate(takes, 1):
                info = stats["joints"][joint]
                if info["uncertainty"] is None:
                    continue
                values.append(info["uncertainty"])
                flag = "" if info["uncertainty"] <= TOLERANCE[joint] else " ⚠"
                print(f"    {name if i == 1 else '':<13}{i:>4}"
                      f"{stats.get('exposure_ms') or 0:>6.2f}ms"
                      f"{stats['brightness']:>6.0f}"
                      f"{info['angle']:>8.1f}°{info['uncertainty']:>9.1f}°"
                      f"{info['noise']:>7.1f}°{info['rate']:>8.0f}°/초{flag}")
            if values:
                print(f"    {'':<13}{'평균':>4}{'':>8}{'':>6}{'':>9}"
                      f"{np.mean(values):>9.1f}°")

    print()
    summary = {}
    for name, takes in results.items():
        values = [s["joints"]["elbow"]["uncertainty"] for s in takes
                  if s["joints"]["elbow"]["uncertainty"] is not None]
        if values:
            summary[name] = (float(np.mean(values)), float(np.max(values)))

    if len(summary) >= 2:
        ranked = sorted(summary.items(), key=lambda kv: kv[1][0])
        print("  팔꿈치 불확실성 (작을수록 좋음):")
        for name, (mean, worst) in ranked:
            print(f"    {name:<10} 평균 ±{mean:.1f}°  최악 ±{worst:.1f}°")
        best, other = ranked[0], ranked[-1]
        if best[1][0] < other[1][0] * 0.7:
            print(f"  → {best[0]}이 뚜렷하게 낫습니다.")
        else:
            print("  → 차이가 뚜렷하지 않습니다. 회차별 값이 서로 겹치는지 보세요.")
        if best[1][1] <= TOLERANCE["elbow"]:
            print(f"  → {best[0]}은 최악의 회차에서도 α=3° 안에 들어왔습니다.")
        else:
            print(f"  → 가장 좋은 설정도 최악 회차가 α=3°를 넘습니다 "
                  "(허용오차 재검토나 한계 서술이 필요).")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    joint_set = LANDMARK_SETS[SIDE]
    total = len(SETTINGS) * REPEATS

    head(f"[준비] {SIDE_LABEL[SIDE]} 기준 — 설정 {len(SETTINGS)}가지 × "
         f"{REPEATS}회 = 스윙 {total}번")
    try:
        cam = RealSenseCamera().open()
    except CameraError as exc:
        print(f"[오류] {exc}")
        sys.exit(1)

    results = {}
    try:
        for option, label in ((rs.option.exposure, "노출"),
                              (rs.option.gain, "감도")):
            rng = cam.color_option_range(option)
            if rng:
                extra = ""
                if option == rs.option.exposure:
                    unit = cam.COLOR_EXPOSURE_UNIT_US / 1000.0
                    extra = f"  → 기본 {rng['default'] * unit:.1f}ms"
                print(f"  컬러 {label} 범위: {rng['min']:.0f}~{rng['max']:.0f}"
                      f"{extra}")
        cam.set_power_line_frequency(60)   # 한국 전원 60Hz
        print()
        print("  매번 빨간 테두리가 되면 바로 스윙하고, 끝나면 멈춰 계세요.")
        print("  스윙 세기는 매번 비슷하게 해주시면 비교가 정확해집니다.")

        aborted = False
        with PoseExtractor() as extractor:
            for name, kind, value in SETTINGS:
                if kind == "auto":
                    cam.set_auto_exposure()
                elif kind == "auto_limit":
                    cam.set_auto_exposure(limit_ms=value)
                else:
                    cam.set_manual_exposure(value)
                # 설정이 실제로 반영되도록 몇 프레임 흘려보낸다
                for _ in range(20):
                    cam.read()

                results[name] = []
                for take in range(1, REPEATS + 1):
                    head(f"[{name}] {take}/{REPEATS}번째 스윙")
                    guide = [f"[{name}] {take}/{REPEATS} — 빨간 테두리에 바로 스윙",
                             "   스윙 뒤에는 그대로 멈춰 계세요"]
                    if not countdown_with_preview(cam, extractor, joint_set,
                                                  SIDE, COUNTDOWN_SECONDS,
                                                  guide):
                        print("\n[중단] 사용자가 중단했습니다.")
                        aborted = True
                        break
                    print("  촬영 중! 스윙하세요!", flush=True)
                    show_recording(cam, extractor, joint_set, SIDE, guide)
                    clip = cam.record(seconds=RECORD_SECONDS)
                    print("  촬영 끝. 분석 중...", flush=True)

                    exposure_now = cam.get_color_exposure_ms()
                    gain_now = cam.get_color_gain()
                    records = build_records(cam, extractor, clip)
                    if len(records) < 10:
                        print(f"  [주의] 사람이 잡힌 프레임이 {len(records)}장뿐 "
                              "— 건너뜁니다.")
                        continue
                    key = f"{name.replace(' ', '_')}_{take}"
                    save_raw(OUTPUT_DIR / f"raw_exp_{key}.npz", records, cam)
                    stats = measure(records, cam, joint_set)
                    if stats:
                        stats["exposure_ms"] = exposure_now
                        stats["gain"] = gain_now
                        results[name].append(stats)
                        elbow = stats["joints"]["elbow"]
                        print(f"  실제 노출 {exposure_now:.2f}ms / 감도 "
                              f"{gain_now:.0f}, 밝기 {stats['brightness']:.0f}, "
                              f"임팩트 {stats['impact']}번, "
                              f"팔꿈치 {elbow['angle']:.1f}° "
                              f"±{elbow['uncertainty']:.1f}° "
                              f"(변화 {elbow['rate']:.0f}°/초)")
                if aborted:
                    break
    finally:
        # 다음 실행에 영향이 가지 않게 자동노출로 되돌린다
        try:
            cam.set_auto_exposure()
        except Exception:
            pass
        cv2.destroyAllWindows()
        cam.close()

    if any(results.values()):
        report({k: v for k, v in results.items() if v})
    else:
        print("[오류] 비교할 결과가 없습니다.")


if __name__ == "__main__":
    main()
