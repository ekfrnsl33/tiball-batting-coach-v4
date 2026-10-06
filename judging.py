"""
판정 로직 + 레벨업 (명세서 5번, 7번)

카메라도 촬영도 필요 없는 순수 계산이다. 그래서 표로 검증할 수 있다
(test_judge.py).

명세서 5번 — 판정:
  측정 각도가 기준각에서 얼마나 벗어났는지 보고 3등급을 준다.
  팔꿈치와 뒷무릎은 각각 독립적으로 판정하며 하나로 뭉뚱그리지 않는다.

명세서 7번 — 레벨업:
  사용 횟수가 아니라 수행 성취도에 연동된다.
  1단계는 허용오차를 3배로 넉넉하게 주고, 잘하면 2배 → 1배로 좁혀간다.
  '완벽' 누적 3회면 승급, '다시 연습' 누적 5회면 강등, 그때 카운트는 0으로.
"""

# ---------------------------------------------------------------------------
# 명세서 5번: 확정된 최종 수치 (이미 관절 사잇각 기준으로 변환된 값)
#
# ⚠️ 이 숫자들은 전문가 그룹 검토를 거친 값이다. 임의로 바꾸지 말 것.
# 문헌·OpenBiomechanics는 "0°=완전 신전" 굴곡각 관례를 쓰지만, 여기 값은
# 이미 180°에서 뺀 관절 사잇각(180°=완전히 폄)이다.
# ---------------------------------------------------------------------------
THETA = {
    "elbow": 100.5,   # 팔꿈치 기준각
    "knee": 109.6,    # 뒷무릎 기준각
}
ALPHA = {"elbow": 3, "knee": 5}     # '완벽' 허용오차
BETA = {"elbow": 6, "knee": 10}     # '좋음' 허용오차

JOINT_LABEL = {"elbow": "팔꿈치", "knee": "뒷무릎"}

# ---------------------------------------------------------------------------
# 명세서 7번: 단계별 허용오차 배율과 승급·강등 조건
# ---------------------------------------------------------------------------
LEVEL_MULTIPLIER = {1: 3, 2: 2, 3: 1}
MIN_LEVEL, MAX_LEVEL = 1, 3
PROMOTE_AFTER = 3     # '완벽' 누적 3회 → 승급
DEMOTE_AFTER = 5      # '다시 연습' 누적 5회 → 강등

PERFECT = "완벽"
GOOD = "좋음"
RETRY = "다시 연습"

# 등급의 높낮이. 세션 등급을 정할 때 쓴다.
GRADE_RANK = {RETRY: 0, GOOD: 1, PERFECT: 2}

# ---------------------------------------------------------------------------
# 명세서 10번 미확정 항목: 레벨업 카운트를 관절별로 분리할지 통합할지
#
# 명세서 권장대로 **통합**으로 구현한다. 통합의 뜻을 정해야 하는데,
# 여기서는 "두 관절 중 더 낮은 등급을 그 세션의 등급으로 본다"로 했다.
#   - 두 관절 모두 '완벽'이어야 '완벽' 1회로 센다
#   - 하나라도 '다시 연습'이면 '다시 연습' 1회로 센다
# 이유: 한쪽만 잘해도 승급하면 나머지 관절을 익히지 않고 넘어가게 된다.
# ⚠️ 이 정의는 사용자 확인이 필요한 부분이다.
# ---------------------------------------------------------------------------


def tolerances(joint, level):
    """그 단계에서 실제로 적용되는 허용오차 (α, β)."""
    multiplier = LEVEL_MULTIPLIER[level]
    return ALPHA[joint] * multiplier, BETA[joint] * multiplier


def judge(measured_angle, theta, alpha, beta):
    """명세서 5번의 판정 함수 그대로."""
    error = abs(measured_angle - theta)
    if error <= alpha:
        return PERFECT
    elif error <= beta:
        return GOOD
    else:
        return RETRY


def judge_joint(joint, measured_angle, level):
    """
    관절 하나를 판정한다.
    반환: {joint, label, angle, theta, error, alpha, beta, grade}
    """
    theta = THETA[joint]
    alpha, beta = tolerances(joint, level)
    return {
        "joint": joint,
        "label": JOINT_LABEL[joint],
        "angle": measured_angle,
        "theta": theta,
        "error": abs(measured_angle - theta),
        "alpha": alpha,
        "beta": beta,
        "grade": judge(measured_angle, theta, alpha, beta),
    }


def judge_session(angles, level):
    """
    한 세션(= 스윙 여러 번의 중앙값)을 판정한다.

    angles: {"elbow": 각도, "knee": 각도}  — 못 구한 관절은 빼고 넘긴다
    반환: (관절별 판정 목록, 세션 등급)

    세션 등급은 두 관절 중 더 낮은 등급이다 (위 주석 참고).
    측정된 관절이 하나도 없으면 등급은 None.
    """
    results = [judge_joint(joint, angle, level)
               for joint, angle in angles.items() if angle is not None]
    if not results:
        return [], None
    worst = min(results, key=lambda r: GRADE_RANK[r["grade"]])
    return results, worst["grade"]


def apply_grade(state, grade):
    """
    판정 결과를 학생 상태에 반영한다.

    state: {"level", "perfect_count", "retry_count"}
    반환: (새 상태, 사건)  사건은 "승급" / "강등" / None

    명세서 7번: 승급·강등이 일어나면 누적 카운트를 0으로 초기화한다.
    '좋음'은 카운트에 영향을 주지 않는다.
    """
    level = int(state.get("level", MIN_LEVEL))
    perfect = int(state.get("perfect_count", 0))
    retry = int(state.get("retry_count", 0))
    event = None

    if grade == PERFECT:
        perfect += 1
        if perfect >= PROMOTE_AFTER and level < MAX_LEVEL:
            level += 1
            perfect = retry = 0
            event = "승급"
    elif grade == RETRY:
        retry += 1
        if retry >= DEMOTE_AFTER and level > MIN_LEVEL:
            level -= 1
            perfect = retry = 0
            event = "강등"

    return ({"level": level, "perfect_count": perfect,
             "retry_count": retry}, event)


def _particle(word):
    """받침이 있으면 '을', 없으면 '를'. (뒷무릎을 / 팔꿈치를)"""
    last = word[-1]
    code = ord(last) - 0xAC00
    if 0 <= code <= 11171:
        return "를" if code % 28 == 0 else "을"
    return "을"


# ---------------------------------------------------------------------------
# v3 요청 1번: 스윙별 개별 점수제
#
# ⚠️ 이 점수는 화면에 추가로 보여주는 정보일 뿐이다. 판정 등급(PERFECT/GOOD/
# RETRY)과 레벨업 로직은 위 함수들 그대로 쓰고, 점수는 등급에서 파생만 된다.
# ---------------------------------------------------------------------------
SCORE = {RETRY: 10, GOOD: 30, PERFECT: 50}


def score_for(grade):
    """등급 하나를 점수로 바꾼다. 등급을 모르면(측정 실패) 0점."""
    return SCORE.get(grade, 0)


# ---------------------------------------------------------------------------
# v3 요청 2번: 방향성 행동 지시어
#
# 오차의 방향(기준각보다 큰지/작은지)에 따라 무엇을 더 하면 될지 말해준다.
# "over" = 측정각이 기준각보다 큼 (팔꿈치는 과신전, 뒷무릎은 덜 굽힘)
# "under" = 측정각이 기준각보다 작음 (팔꿈치는 과굴곡, 뒷무릎은 과굴곡)
#
# v4 요청 1번: 방향은 그대로 두고, 오차 "크기"에 따른 정량적 수식어를 문구에
# 끼워 넣는다. 새 임계값을 만들지 않고 기존 β(좋음 허용오차)를 그대로
# 재사용한다 — 오차 ≤ 2β면 "조금", 2β 초과면 "많이". (완벽 등급은 애초에
# 이 함수를 호출하지 않으므로 여기서 다루는 오차는 항상 α보다 크다.)
# 관절별 비유 단위는 전문가 제안(주먹/뼘)을 그대로 썼다.
# ---------------------------------------------------------------------------
MAGNITUDE_UNIT = {
    "elbow": {"조금": "주먹 하나만큼", "많이": "주먹 두 개만큼"},
    "knee": {"조금": "한 뼘만큼", "많이": "두 뼘만큼"},
}

DIRECTIONAL_TEMPLATES = {
    "elbow": {
        "over": ["팔꿈치를 몸쪽으로 {unit} 더 접어볼까?",
                "배트를 몸 쪽으로 {unit} 더 붙여서 휘둘러보자"],
        "under": ["팔을 {unit} 더 펴볼까?", "끝까지 팔을 {unit} 더 뻗어보자"],
    },
    "knee": {
        "over": ["뒷무릎을 {unit} 더 굽혀볼까?",
                "한 발 뒤로 서서 중심을 {unit} 더 낮춰보자"],
        "under": ["무릎을 {unit} 더 펴볼까?"],
    },
}


def magnitude_level(error, beta):
    """오차 ≤ 2β면 '조금', 2β 초과면 '많이'. (β는 그 단계에서 적용된 좋음 허용오차)"""
    return "조금" if error <= 2 * beta else "많이"


def direction_phrases(joint, angle, theta, beta):
    """그 관절·그 오차 방향·크기에 맞는 행동 지시어 후보 목록을 돌려준다."""
    key = "over" if angle > theta else "under"
    level = magnitude_level(abs(angle - theta), beta)
    unit = MAGNITUDE_UNIT[joint][level]
    return [template.format(unit=unit)
           for template in DIRECTIONAL_TEMPLATES[joint][key]]


def feedback(grade, results):
    """
    명세서 8번의 피드백 문구. 등급별로 격려 톤을 유지한다.
    '다시 연습'도 실패 지적이 아니라 재도전 독려로 쓴다.
    """
    if grade == PERFECT:
        return "완벽해요! 지금 자세를 그대로 기억해 두세요."
    if grade == GOOD:
        # 어느 관절을 다듬어야 하는지 한 가지만 콕 집어준다
        target = max(results, key=lambda r: r["error"] / max(r["beta"], 1e-9))
        direction = "조금 더 펴" if target["angle"] < target["theta"] else "조금 더 굽혀"
        label = target["label"]
        return (f"좋아요! {label}{_particle(label)} {direction} 보면 "
                "더 좋아질 거예요.")
    if grade == RETRY:
        return "괜찮아요, 한 번 더 해볼까요? 천천히 자세를 잡아 보세요."
    return "측정이 잘 되지 않았어요. 다시 한 번 촬영해 볼까요?"
