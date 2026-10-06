"""
[4단계] 한 세션 전체 흐름 — 촬영 → 임팩트 → 각도 → 판정 → 레벨 갱신 → 기록

명세서 대응:
  2번  2~4초 단기 녹화 (3초 사용)
  4번  각도 산출 (3D 좌표, 명세서 검증 후 확정)
  5번  판정 (관절별 3등급)
  6번  임팩트 프레임 = 손목 최고 속도
  7번  레벨업 상태 관리 (학생별 저장)
  8번  피드백 문구, 최근 이력
  9번  CSV 기록, 브라우저용 골격선 전용 이미지

**한 판정에 스윙을 여러 번 찍는 이유** (2026-09-05 결정):
  팔꿈치 각도의 측정 불확실성이 ±5.2°로, 명세서 5번의 α=3°보다 크다.
  스윙 한 번으로는 3단계에서 '완벽'과 '좋음'을 가를 수 없다.
  스윙을 n번 찍어 중앙값을 쓰면 불확실성이 대략 1/√n로 줄어든다.
  3번이면 ±3° 수준이 되어 판정이 성립한다.
  남는 불확실성은 논문에 한계로 서술하기로 했다.

실행 (관리자 권한 필요):
  cd ~/Desktop/tiball_project
  sudo ./venv/bin/python -u test_session.py --student S1 --side left
"""

import sys
from pathlib import Path

import cv2
import numpy as np

import impact as impact_mod
import storage
from angles import ANGLE_WINDOW_HALF, angle_fit_at, point_from_record
from camera import RealSenseCamera, CameraError
from capture import (SIDE_LABEL, build_records, countdown_with_preview,
                     save_raw, show_recording, side_from_args)
from judging import (JOINT_LABEL, LEVEL_MULTIPLIER, apply_grade, feedback,
                     judge_session, tolerances)
from pose import LANDMARK_SETS, PoseExtractor, draw_skeleton, skeleton_only

OUTPUT_DIR = Path(__file__).parent / "test_output"

RECORD_SECONDS = 3.0
COUNTDOWN_SECONDS = 8
SWINGS_PER_SESSION = 3      # 위 주석 참고

JOINTS = {
    "elbow": ("shoulder", "elbow", "wrist"),
    "knee": ("hip", "knee", "ankle"),
}


def head(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def student_from_args(argv, default="S1"):
    for i, arg in enumerate(argv):
        if arg in ("--student", "-p") and i + 1 < len(argv):
            return argv[i + 1]
    return default


def analyze_swing(records, cam, joint_set):
    """스윙 한 번에서 임팩트 각도와 불확실성을 뽑는다."""
    wrists = [LANDMARK_SETS["left"]["wrist"], LANDMARK_SETS["right"]["wrist"]]
    found = impact_mod.find_impact(records, wrists, cam, cam.fps, "3D(m)")
    if found is None:
        return None

    peak = found["index"]
    out = {"impact": peak, "competing": len(found["competing"]),
           "peak_speed": found["peak_speed"], "joints": {}, "coords": {}}
    for joint, roles in JOINTS.items():
        indices = [joint_set[role] for role in roles]
        fit = angle_fit_at(records, indices, cam, peak)
        out["joints"][joint] = fit
        # 명세서 9번: 관절 좌표값도 기록에 남긴다 (임팩트 프레임의 꼭짓점)
        out["coords"][joint] = point_from_record(records[peak], indices[1], cam)
    return out


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    side = side_from_args(sys.argv)
    student = student_from_args(sys.argv)
    joint_set = LANDMARK_SETS[side]

    state = storage.get_state(student)
    level = state["level"]

    head(f"[준비] 학생 {student} — {SIDE_LABEL[side]}")
    print(f"  현재 단계: {level}단계 (허용오차 ×{LEVEL_MULTIPLIER[level]})")
    print(f"  누적: 완벽 {state['perfect_count']}회 / "
          f"다시연습 {state['retry_count']}회")
    for joint in JOINTS:
        alpha, beta = tolerances(joint, level)
        print(f"    {JOINT_LABEL[joint]}: 완벽 ±{alpha}° / 좋음 ±{beta}°")
    print()
    print(f"  이번 판정은 스윙 {SWINGS_PER_SESSION}번의 중앙값으로 합니다.")

    try:
        cam = RealSenseCamera().open()
    except CameraError as exc:
        print(f"[오류] {exc}")
        sys.exit(1)

    swings = []
    try:
        with PoseExtractor() as extractor:
            for take in range(1, SWINGS_PER_SESSION + 1):
                head(f"[촬영] {take}/{SWINGS_PER_SESSION}번째 스윙")
                guide = [f"{take}/{SWINGS_PER_SESSION}번째 — 빨간 테두리에 스윙",
                         "   스윙 뒤에는 그대로 멈춰 계세요"]
                if not countdown_with_preview(cam, extractor, joint_set, side,
                                              COUNTDOWN_SECONDS, guide):
                    print("\n[중단] 사용자가 중단했습니다.")
                    break
                print("  촬영 중! 스윙하세요!", flush=True)
                show_recording(cam, extractor, joint_set, side, guide)
                clip = cam.record(seconds=RECORD_SECONDS)
                print("  분석 중...", flush=True)

                records = build_records(cam, extractor, clip)
                if len(records) < 10:
                    print(f"  [주의] 사람이 잡힌 프레임이 {len(records)}장뿐 "
                          "— 이 스윙은 버립니다.")
                    continue
                result = analyze_swing(records, cam, joint_set)
                if result is None:
                    print("  [주의] 임팩트를 찾지 못했습니다 — 이 스윙은 버립니다.")
                    continue

                result["records"] = records
                swings.append(result)
                save_raw(OUTPUT_DIR / f"raw_session_{take}.npz", records, cam)

                parts = []
                for joint in JOINTS:
                    fit = result["joints"][joint]
                    parts.append(f"{JOINT_LABEL[joint]} "
                                 + (f"{fit['angle']:.1f}°" if fit
                                    else "측정불가"))
                warn = "  ⚠ 스윙 외 동작 섞임" if result["competing"] else ""
                print(f"  임팩트 {result['impact']}번, " + ", ".join(parts)
                      + warn)
    finally:
        cv2.destroyAllWindows()
        cam.close()

    if not swings:
        print("\n[오류] 쓸 수 있는 스윙이 없습니다. 다시 촬영해주세요.")
        sys.exit(2)

    # -----------------------------------------------------------------------
    head(f"[집계] 스윙 {len(swings)}번의 중앙값")
    angles, uncertainties = {}, {}
    columns = "".join(f"{str(i) + '회차':>12}"
                      for i in range(1, len(swings) + 1))
    print(f"  {'관절':<7}{columns}{'중앙값':>10}{'스윙간SD':>10}"
          f"{'판정 불확실성':>15}{'측정잡음':>10}")
    print("  " + "-" * (7 + 12 * len(swings) + 45))
    for joint in JOINTS:
        values, uncs = [], []
        for swing in swings:
            fit = swing["joints"][joint]
            if fit and fit["angle"] is not None:
                values.append(fit["angle"])
                if fit["uncertainty"] is not None:
                    uncs.append(fit["uncertainty"])
        if not values:
            angles[joint] = None
            print(f"  {JOINT_LABEL[joint]:<7}" + "측정불가".rjust(12))
            continue
        median = float(np.median(values))

        # 판정에 쓰는 값(중앙값)의 불확실성은 두 가지가 함께 들어간다.
        #   (1) 한 스윙 안의 측정 잡음
        #   (2) 스윙마다 자세가 실제로 달라지는 편차
        # 스윙 간 표준오차(SD/√n)는 이미 (1)과 (2)를 모두 포함하므로
        # 그것을 판정 불확실성으로 쓴다. 측정 잡음만 쓰면 실제보다
        # 훨씬 낙관적인 숫자가 된다 (2026-09-05 실측에서 3.2° vs 6.1°).
        noise = float(np.mean(uncs)) if uncs else None
        if len(values) >= 2:
            spread = float(np.std(values, ddof=1))
            combined = spread / np.sqrt(len(values))
        else:
            spread, combined = None, noise

        angles[joint] = median
        uncertainties[joint] = combined
        cells = "".join(f"{v:11.1f}°" for v in values)
        print(f"  {JOINT_LABEL[joint]:<7}{cells}{median:9.1f}°"
              f"{spread if spread is not None else 0:9.1f}°"
              f"{combined:14.1f}°{noise if noise else 0:9.1f}°")

    # -----------------------------------------------------------------------
    head("[판정] 명세서 5번")
    results, grade = judge_session(angles, level)
    print(f"  {'관절':<7}{'측정':>9}{'기준각':>9}{'오차':>8}"
          f"{'완벽까지':>10}{'등급':>10}")
    print("  " + "-" * 56)
    for r in results:
        print(f"  {r['label']:<7}{r['angle']:>8.1f}°{r['theta']:>8.1f}°"
              f"{r['error']:>7.1f}°{r['alpha']:>9}°{r['grade']:>10}")
    print()
    print(f"  세션 등급: {grade}   (두 관절 중 낮은 등급)")
    print(f"  피드백: {feedback(grade, results)}")

    # 판정 불확실성이 허용오차보다 크면 그 등급을 믿기 어렵다.
    # 경계에 가까운 경우에만 경고한다.
    print()
    for r in results:
        unc = uncertainties.get(r["joint"])
        if unc is None:
            continue
        margin_alpha = abs(r["error"] - r["alpha"])
        margin_beta = abs(r["error"] - r["beta"])
        if min(margin_alpha, margin_beta) < unc:
            print(f"  ⚠ {r['label']}: 오차 {r['error']:.1f}°가 등급 경계"
                  f"(±{r['alpha']}°/±{r['beta']}°)에서 불확실성"
                  f"({unc:.1f}°)보다 가깝습니다 → 등급이 바뀔 수 있습니다.")
        elif unc > r["alpha"]:
            print(f"  · {r['label']}: 불확실성 {unc:.1f}°가 '완벽' 기준"
                  f"(±{r['alpha']}°)보다 큽니다. 지금 등급은 경계에서 멀어"
                  "판정 자체는 안전합니다.")

    # -----------------------------------------------------------------------
    head("[레벨] 명세서 7번")
    before = dict(state)
    after, event = apply_grade(state, grade)
    storage.set_state(student, after)
    print(f"  {before['level']}단계 (완벽 {before['perfect_count']} / "
          f"다시연습 {before['retry_count']})"
          f"  →  {after['level']}단계 (완벽 {after['perfect_count']} / "
          f"다시연습 {after['retry_count']})")
    if event:
        print(f"  ★ {event}!")
    else:
        need = 3 - after["perfect_count"]
        print(f"  다음 승급까지 '완벽' {need}회 남음")

    # -----------------------------------------------------------------------
    head("[기록] 명세서 9번")
    best = swings[len(swings) // 2]
    row = {
        "timestamp": storage.now_text(),
        "student_id": student,
        "batting_side": side,
        "level_before": before["level"],
        "level_after": after["level"],
        "perfect_count": after["perfect_count"],
        "retry_count": after["retry_count"],
        "event": event or "",
        "swings": len(swings),
        "session_grade": grade,
        "impact_frame": best["impact"],
        "joint_coords": storage.format_coords(best["coords"]),
    }
    for r in results:
        row[f"{r['joint']}_angle"] = round(r["angle"], 2)
        row[f"{r['joint']}_grade"] = r["grade"]
        unc = uncertainties.get(r["joint"])
        row[f"{r['joint']}_uncertainty"] = (round(unc, 2) if unc is not None
                                            else "")
    _shown, _skeleton = storage.result_paths(student, row["timestamp"])
    root = Path(__file__).parent
    row["image_shown"] = str(_shown.relative_to(root))
    row["image_skeleton"] = str(_skeleton.relative_to(root))
    path = storage.append_session(row)
    print(f"  세션 기록: {path}")
    print(f"  학생 상태: {storage.STUDENTS_FILE}")

    # 결과 이미지 두 장을 학생별 폴더에 남긴다 (app.py와 같은 규칙)
    record = best["records"][best["impact"]]
    highlight = sorted(set(joint_set.values()))
    shown = draw_skeleton(record["color"], record["landmarks"],
                          highlight_indices=highlight)
    skeleton = skeleton_only(record["color"].shape, record["landmarks"],
                             highlight_indices=highlight)
    shown_path, skeleton_path = storage.result_paths(student, row["timestamp"])
    cv2.imwrite(str(shown_path), shown)
    cv2.imwrite(str(skeleton_path), skeleton)
    storage.fix_permissions(shown_path)
    storage.fix_permissions(skeleton_path)
    print(f"  학생 모습 + 골격선: {shown_path.relative_to(Path(__file__).parent)}")
    print(f"  골격선만: {skeleton_path.relative_to(Path(__file__).parent)}")

    # -----------------------------------------------------------------------
    head("[최근 이력] 명세서 8번 사이드 패널에 표시할 내용")
    recent = storage.recent_sessions(student, limit=5)
    if not recent:
        print("  (기록 없음)")
    else:
        print(f"  {'일시':<21}{'등급':<10}{'팔꿈치':>9}{'뒷무릎':>9}"
              f"{'단계':>7}")
        print("  " + "-" * 58)
        for item in recent:
            print(f"  {item['timestamp']:<21}{item['session_grade']:<10}"
                  f"{item.get('elbow_angle', '-'):>9}"
                  f"{item.get('knee_angle', '-'):>9}"
                  f"{item.get('level_after', '-'):>7}단계")


if __name__ == "__main__":
    main()
