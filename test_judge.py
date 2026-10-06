"""
[4단계 검증] 판정 로직과 레벨업을 표로 확인한다 (명세서 5번, 7번)

카메라가 필요 없다. 숫자를 넣어보고 결과가 명세서대로 나오는지 확인만 한다.
학생 상태 저장도 임시 폴더에서 시험하므로 실제 데이터는 건드리지 않는다.

실행 (관리자 권한 필요 없음):
  cd ~/Desktop/tiball_project
  ./venv/bin/python test_judge.py
"""

import shutil
import tempfile
from pathlib import Path

import judging
import storage
from judging import (ALPHA, BETA, GOOD, LEVEL_MULTIPLIER, PERFECT, RETRY,
                     THETA, apply_grade, feedback, judge_joint, judge_session,
                     tolerances)


def head(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def check(label, got, expected):
    ok = got == expected
    mark = "OK" if ok else "틀림"
    print(f"  {label:<46} {str(got):<12} {mark}")
    if not ok:
        print(f"      ↳ 기대값: {expected}")
    return ok


def main():
    failures = 0

    # -----------------------------------------------------------------------
    head("[1] 단계별 허용오차 (명세서 7번: 1단계 ×3, 2단계 ×2, 3단계 ×1)")
    print(f"  {'단계':<6}{'배율':>5}   팔꿈치 완벽/좋음      뒷무릎 완벽/좋음")
    print("  " + "-" * 58)
    for level in (1, 2, 3):
        ae, be = tolerances("elbow", level)
        ak, bk = tolerances("knee", level)
        print(f"  {level}단계{LEVEL_MULTIPLIER[level]:>5}배   "
              f"±{ae:>2}° / ±{be:>2}°            ±{ak:>2}° / ±{bk:>2}°")
    print()
    print(f"  기준각: 팔꿈치 {THETA['elbow']}° / 뒷무릎 {THETA['knee']}°")
    print(f"  기본 허용오차: 팔꿈치 α={ALPHA['elbow']}° β={BETA['elbow']}° / "
          f"뒷무릎 α={ALPHA['knee']}° β={BETA['knee']}°")

    # -----------------------------------------------------------------------
    head("[2] 경계값 판정 — 3단계(배율 ×1) 팔꿈치, 기준각 100.5°")
    print(f"  {'측정 각도':<12}{'오차':>7}   판정")
    print("  " + "-" * 40)
    cases = [
        (100.5, 0.0, PERFECT),
        (103.5, 3.0, PERFECT),      # α 경계는 '완벽'에 포함
        (103.6, 3.1, GOOD),
        (106.5, 6.0, GOOD),         # β 경계는 '좋음'에 포함
        (106.6, 6.1, RETRY),
        (94.5, 6.0, GOOD),          # 반대 방향도 대칭
        (94.4, 6.1, RETRY),
    ]
    for angle, expected_error, expected_grade in cases:
        result = judge_joint("elbow", angle, level=3)
        line = f"  {angle:<12.1f}{result['error']:>7.1f}   {result['grade']}"
        ok = (result["grade"] == expected_grade
              and abs(result["error"] - expected_error) < 1e-6)
        print(line + ("" if ok else f"   ← 틀림 (기대 {expected_grade})"))
        failures += 0 if ok else 1

    # -----------------------------------------------------------------------
    head("[3] 1단계에서는 같은 각도가 어떻게 판정되나 (허용오차 3배)")
    print(f"  {'측정 각도':<12}{'1단계':>10}{'2단계':>10}{'3단계':>10}")
    print("  " + "-" * 44)
    for angle in (100.5, 105.0, 108.0, 112.0, 120.0):
        grades = [judge_joint("elbow", angle, level)["grade"]
                  for level in (1, 2, 3)]
        print(f"  {angle:<12.1f}" + "".join(f"{g:>10}" for g in grades))
    print()
    print("  → 같은 자세라도 낮은 단계에서는 넉넉하게, 높은 단계에서는")
    print("    엄격하게 판정됩니다. 명세서 7번의 의도대로입니다.")

    # -----------------------------------------------------------------------
    head("[4] 두 관절을 합쳐 세션 등급을 정하는 규칙 (명세서 10번 미확정 항목)")
    print("  구현: 두 관절 중 '더 낮은 등급'을 세션 등급으로 삼음")
    print()
    print(f"  {'팔꿈치':<10}{'뒷무릎':<12}{'세션 등급':<12}")
    print("  " + "-" * 36)
    combos = [
        (100.5, 109.6, PERFECT),      # 둘 다 완벽
        (100.5, 115.0, GOOD),         # 하나는 완벽, 하나는 좋음
        (100.5, 130.0, RETRY),        # 하나라도 다시 연습이면 다시 연습
        (110.0, 109.6, RETRY),
        (104.0, 112.0, GOOD),
    ]
    for elbow, knee, expected in combos:
        results, grade = judge_session({"elbow": elbow, "knee": knee}, level=3)
        names = {r["joint"]: r["grade"] for r in results}
        ok = grade == expected
        print(f"  {names['elbow']:<10}{names['knee']:<12}{grade:<12}"
              + ("" if ok else f"← 틀림 (기대 {expected})"))
        failures += 0 if ok else 1

    # -----------------------------------------------------------------------
    head("[5] 승급 — '완벽' 누적 3회면 다음 단계, 카운트는 0으로")
    state = {"level": 1, "perfect_count": 0, "retry_count": 0}
    print(f"  {'회차':<6}{'판정':<10}{'단계':>5}{'완벽':>6}{'다시연습':>9}   사건")
    print("  " + "-" * 50)
    for i, grade in enumerate([PERFECT, PERFECT, PERFECT, PERFECT], 1):
        state, event = apply_grade(state, grade)
        print(f"  {i:<6}{grade:<10}{state['level']:>5}"
              f"{state['perfect_count']:>6}{state['retry_count']:>9}   "
              f"{event or ''}")
    failures += 0 if check("3회째에 2단계로 승급했는가", state["level"], 2) else 1
    failures += 0 if check("승급 후 완벽 카운트가 1인가 (4회째 적립)",
                           state["perfect_count"], 1) else 1

    # -----------------------------------------------------------------------
    head("[6] 강등 — '다시 연습' 누적 5회면 이전 단계")
    state = {"level": 3, "perfect_count": 2, "retry_count": 0}
    print(f"  {'회차':<6}{'판정':<10}{'단계':>5}{'완벽':>6}{'다시연습':>9}   사건")
    print("  " + "-" * 50)
    for i in range(1, 6):
        state, event = apply_grade(state, RETRY)
        print(f"  {i:<6}{RETRY:<10}{state['level']:>5}"
              f"{state['perfect_count']:>6}{state['retry_count']:>9}   "
              f"{event or ''}")
    failures += 0 if check("5회째에 2단계로 강등됐는가", state["level"], 2) else 1
    failures += 0 if check("강등 시 완벽 카운트도 0으로 초기화됐는가",
                           state["perfect_count"], 0) else 1

    # -----------------------------------------------------------------------
    head("[7] '좋음'은 카운트에 영향을 주지 않는다")
    state = {"level": 2, "perfect_count": 1, "retry_count": 2}
    after, event = apply_grade(dict(state), GOOD)
    failures += 0 if check("단계 유지", after["level"], state["level"]) else 1
    failures += 0 if check("완벽 카운트 유지", after["perfect_count"],
                           state["perfect_count"]) else 1
    failures += 0 if check("다시연습 카운트 유지", after["retry_count"],
                           state["retry_count"]) else 1
    failures += 0 if check("사건 없음", event, None) else 1

    # -----------------------------------------------------------------------
    head("[8] 최상단·최하단에서는 더 오르거나 내려가지 않는다")
    state = {"level": 3, "perfect_count": 2, "retry_count": 0}
    for _ in range(4):
        state, _ = apply_grade(state, PERFECT)
    failures += 0 if check("3단계에서 완벽을 더 해도 3단계 유지",
                           state["level"], 3) else 1
    state = {"level": 1, "perfect_count": 0, "retry_count": 4}
    for _ in range(6):
        state, _ = apply_grade(state, RETRY)
    failures += 0 if check("1단계에서 다시연습을 더 해도 1단계 유지",
                           state["level"], 1) else 1

    # -----------------------------------------------------------------------
    head("[9] 학생별로 상태가 따로 저장되는가 (명세서 7번 전제조건)")
    temp = Path(tempfile.mkdtemp())
    storage.DATA_DIR = temp
    storage.STUDENTS_FILE = temp / "students.json"
    storage.SESSIONS_CSV = temp / "sessions.csv"
    try:
        students = storage.load_students()
        print(f"  기본 학생 목록: {', '.join(students)}")

        # S1은 완벽 3회로 승급, S2는 다시연습만
        state = storage.get_state("S1")
        for _ in range(3):
            state, _ = apply_grade(state, PERFECT)
        storage.set_state("S1", state)

        state = storage.get_state("S2")
        for _ in range(2):
            state, _ = apply_grade(state, RETRY)
        storage.set_state("S2", state)

        # 파일에서 다시 읽어서 확인 (프로그램을 껐다 켠 상황)
        s1, s2, s3 = (storage.get_state(k) for k in ("S1", "S2", "S3"))
        print(f"  S1 → {s1}")
        print(f"  S2 → {s2}")
        print(f"  S3 → {s3}  (건드리지 않은 학생)")
        failures += 0 if check("S1은 2단계로 승급", s1["level"], 2) else 1
        failures += 0 if check("S2는 1단계 유지, 다시연습 2회",
                               (s2["level"], s2["retry_count"]), (1, 2)) else 1
        failures += 0 if check("S3은 초기 상태 그대로",
                               s3, {"level": 1, "perfect_count": 0,
                                    "retry_count": 0}) else 1

        # 세션 기록 CSV
        head("[10] 세션 기록 CSV (명세서 9번)")
        row = {
            "timestamp": storage.now_text(),
            "student_id": "S1",
            "batting_side": "left",
            "level_before": 1, "level_after": 2,
            "perfect_count": 0, "retry_count": 0, "event": "승급",
            "swings": 3,
            "elbow_angle": 101.2, "elbow_grade": PERFECT,
            "elbow_uncertainty": 5.2,
            "knee_angle": 111.0, "knee_grade": PERFECT,
            "knee_uncertainty": 2.1,
            "session_grade": PERFECT, "impact_frame": 65,
            "joint_coords": storage.format_coords(
                {"elbow": (0.12, -0.34, 2.51), "knee": (0.08, 0.41, 2.63)}),
        }
        path = storage.append_session(row)
        print(f"  저장: {path.name}")
        recent = storage.recent_sessions("S1")
        failures += 0 if check("방금 기록이 조회되는가", len(recent), 1) else 1
        if recent:
            print(f"  최근 이력 1건: {recent[0]['timestamp']} "
                  f"{recent[0]['session_grade']} "
                  f"(팔꿈치 {recent[0]['elbow_angle']}°, "
                  f"뒷무릎 {recent[0]['knee_angle']}°)")
            print(f"  좌표 기록: {recent[0]['joint_coords']}")
            failures += 0 if check("실명이 저장되지 않았는가 (학생ID만)",
                                   recent[0]["student_id"], "S1") else 1
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    # -----------------------------------------------------------------------
    head("[11] 피드백 문구 (명세서 8번: 등급별 격려 톤)")
    for elbow, knee in ((100.5, 109.6), (100.5, 116.0), (120.0, 130.0)):
        results, grade = judge_session({"elbow": elbow, "knee": knee}, level=3)
        print(f"  [{grade}] {feedback(grade, results)}")

    # -----------------------------------------------------------------------
    head("검증 결과")
    if failures == 0:
        print("  모든 항목 통과.")
    else:
        print(f"  ⚠ {failures}개 항목이 기대와 다릅니다.")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
