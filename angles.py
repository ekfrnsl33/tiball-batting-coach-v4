"""
각도 계산 모듈 (명세서 4번)

세 점 A(근위) - B(꼭짓점) - C(원위)의 사잇각을 구한다.

    v1 = A - B
    v2 = C - B
    θ = arccos( (v1·v2) / (|v1| × |v2|) )

명세서 4번 규약: **θ = 180°일 때 완전히 폄, 0°에 가까울수록 굽음**
(문헌·OpenBiomechanics의 "0°=완전 신전" 굴곡각과는 정반대 방향이므로 주의)

좌표를 만드는 방식이 세 가지 있어서 결과가 달라진다. 어느 쪽이 맞는지
확인하려고 세 가지를 다 만들어 뒀다 (test_angle.py에서 비교).

  as_2d    : 화면 좌표만 (심도 안 씀)
  as_mixed : 명세서 2번 표기 그대로 P=(x, y, d) — x,y는 픽셀, d는 미터
  as_3d    : 렌즈 정보로 세 축 모두 미터로 환산한 실제 3차원 좌표
"""

import numpy as np

import depth_sampling


def joint_angle(a, b, c):
    """
    세 점의 사잇각(도). b가 꼭짓점.
    점은 2차원이든 3차원이든 상관없지만, 세 점의 차원은 같아야 한다.
    """
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None

    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    # 명세서 4번: 부동소수점 오차 때문에 1.0000001 같은 값이 나올 수 있어
    # arccos가 터진다. 반드시 [-1, 1]로 잘라줘야 한다.
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


# ---------------------------------------------------------------------------
# 좌표를 만드는 세 가지 방식
# ---------------------------------------------------------------------------

def as_2d(landmark, depth_m=None, camera=None):
    """화면 좌표만 사용 (심도 무시)."""
    return (landmark["x"], landmark["y"])


def as_mixed(landmark, depth_m, camera=None):
    """명세서 2번 표기 그대로 P = (x, y, d). x, y는 픽셀 / d는 미터."""
    if depth_m is None:
        return None
    return (landmark["x"], landmark["y"], depth_m)


def as_3d(landmark, depth_m, camera):
    """렌즈 정보로 세 축 모두 미터로 환산한 실제 3차원 좌표."""
    if depth_m is None:
        return None
    return camera.pixel_to_point(landmark["x"], landmark["y"], depth_m)


COORD_MODES = {
    "2D": as_2d,
    "P=(x,y,d)": as_mixed,
    "3D(m)": as_3d,
}


def angle_from_landmarks(landmarks, indices, depth_image, camera, mode):
    """
    랜드마크 세 개(근위, 꼭짓점, 원위)로 각도를 구한다.
    심도가 필요한 방식인데 심도를 못 읽으면 None.
    """
    build = COORD_MODES[mode]
    points = []
    for idx in indices:
        lm = landmarks[idx]
        depth_m = camera.sample_depth(depth_image, lm["x"], lm["y"])
        point = build(lm, depth_m, camera)
        if point is None:
            return None
        points.append(point)
    return joint_angle(*points)


# ---------------------------------------------------------------------------
# 기록(capture.py의 record)에서 바로 각도를 구하는 함수.
# 3단계 이후 실제 파이프라인이 쓰는 것은 이쪽이다.
# ---------------------------------------------------------------------------

# 2026-09-05 검증으로 확정: 세 축을 모두 미터로 환산하는 방식을 쓴다.
# 명세서 2번의 P=(x,y,d)를 문자 그대로 쓰면 단위가 섞여 심도가 무시된다.
DEFAULT_MODE = "3D(m)"


def point_from_record(record, index, camera, mode=DEFAULT_MODE, sampler=None):
    """기록의 관절 하나를 좌표점으로 만든다. 심도를 못 읽으면 None."""
    depth_m = None
    if mode != "2D":
        depth_m = depth_sampling.sample(
            record["patches"][index], camera.depth_scale,
            sampler or depth_sampling.DEFAULT_SAMPLER)
    landmark = {"x": float(record["x"][index]),
                "y": float(record["y"][index])}
    return COORD_MODES[mode](landmark, depth_m, camera)


def angle_from_record(record, indices, camera, mode=DEFAULT_MODE,
                      sampler=None):
    """기록 한 장에서 관절 각도(도)를 구한다. 못 구하면 None."""
    points = []
    for index in indices:
        point = point_from_record(record, index, camera, mode, sampler)
        if point is None:
            return None
        points.append(point)
    return joint_angle(*points)


# 한 프레임의 각도는 잡음이 크다 (2026-09-05 실측: 팔꿈치 프레임간 변화가
# 중앙값 4.9°, 상위 10%는 17°, 최대 99.6°). 임팩트 한 장만 쓰면 하필 튄
# 프레임을 집을 수 있으므로, 그 주변 여러 프레임을 함께 본다.
#
# 창 크기는 실측으로 정했다 (스윙 6회, 창 크기별 팔꿈치 불확실성):
#   ±3 → 9.1°   ±5 → 6.6°   ±7 → 5.2°   ±9 → 6.0°   ±13 → 6.5°
# 넓히면 잡음이 평균되어 줄지만, 너무 넓히면 각도 변화가 직선을 벗어나
# 오히려 나빠진다. ±7프레임(60fps에서 ±117ms)이 최적이었다.
# 2차식도 시험했으나 팔꿈치에서 스윙 간 편차가 커져(과적합) 1차식을 쓴다.
ANGLE_WINDOW_HALF = 7


def angle_around(records, indices, camera, center, half=ANGLE_WINDOW_HALF,
                 mode=DEFAULT_MODE, sampler=None):
    """
    center 프레임 주변(±half)의 각도 중앙값과 그 흔들림을 돌려준다.
    반환: (중앙값, 표준편차, 사용한 프레임 수) — 못 구하면 (None, None, 0)
    """
    values = []
    lo = max(0, center - half)
    hi = min(len(records), center + half + 1)
    for i in range(lo, hi):
        angle = angle_from_record(records[i], indices, camera, mode, sampler)
        if angle is not None:
            values.append(angle)
    if not values:
        return None, None, 0
    return float(np.median(values)), float(np.std(values)), len(values)


def angle_fit_at(records, indices, camera, center, half=ANGLE_WINDOW_HALF,
                 mode=DEFAULT_MODE, sampler=None):
    """
    임팩트 프레임의 각도를 '주변 창에 직선을 맞춰' 추정한다.

    왜 이렇게 하는가 (2026-09-05 실측 근거):
      임팩트 순간 팔꿈치는 초당 300~600°로 빠르게 펴지는 중이다. 그래서
      주변 프레임의 단순 중앙값을 쓰면 실제 변화까지 뭉개져 값이 치우친다.
      반대로 한 프레임만 쓰면 잡음(±5~25°)을 그대로 맞는다.

      주변 창에 직선을 맞추면 '실제로 변하는 추세'는 직선이 흡수하고,
      직선에서 벗어난 정도가 곧 잡음이 된다. 그 직선을 임팩트 프레임에서
      읽으면 잡음이 프레임 수만큼 평균되어 줄어든다 (대략 잡음/√n).

    반환 dict (못 구하면 None):
      angle        : 임팩트 프레임의 추정 각도
      noise        : 잡음 크기 (직선에서 벗어난 정도, RMS)
      uncertainty  : 추정값의 불확실성 (noise / √프레임수)
      rate         : 그 순간 각도가 변하는 속도 (도/초)
      used         : 사용한 프레임 수
    """
    lo = max(0, center - half)
    hi = min(len(records), center + half + 1)
    frames, values = [], []
    for i in range(lo, hi):
        angle = angle_from_record(records[i], indices, camera, mode, sampler)
        if angle is not None:
            frames.append(i)
            values.append(angle)
    if len(values) < 4:
        # 직선을 맞출 만큼 없으면 한 프레임 값으로 대신한다
        single = angle_from_record(records[center], indices, camera, mode,
                                   sampler)
        if single is None:
            return None
        return {"angle": single, "noise": None, "uncertainty": None,
                "rate": None, "used": 1}

    frames = np.asarray(frames, dtype=float)
    values = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(frames, values, 1)
    residual = values - (slope * frames + intercept)
    noise = float(np.sqrt(np.mean(residual ** 2)))
    return {
        "angle": float(slope * center + intercept),
        "noise": noise,
        "uncertainty": noise / np.sqrt(len(values)),
        "rate": float(slope),          # 프레임당 변화 (초당 값은 호출자가 환산)
        "used": len(values),
    }
