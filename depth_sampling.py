"""
관절 위치에서 '거리'를 읽어내는 방법들

왜 여러 방법이 필요한가:
  관절 점(예: 손목) 주변의 작은 사각형 안에는 여러 거리가 섞여 있다.
  손이 몸통 앞에 있으면 그 사각형에 '손의 거리'와 '몸통의 거리'가 함께
  들어온다. 어느 값을 고르느냐에 따라 각도가 30도 이상 달라진다.

  지금까지는 중앙값을 썼는데, 그러면 손과 몸통이 섞여 엉뚱한 값이 나온다.
  팔다리는 그 뒤에 있는 것보다 항상 카메라에 더 가깝다는 점을 이용하면,
  '가까운 쪽'을 고르는 게 맞다.

각 방법:
  중앙값   — 주변 값의 중앙값. 잡음에 강하지만 앞뒤가 섞이면 틀린다 (기존 방식)
  최소     — 가장 가까운 값. 앞면을 확실히 잡지만 잡음 한 점에 흔들린다
  하위10%  — 가까운 쪽 10% 지점. 최소값의 잡음 문제를 줄인 것
  앞면     — 가장 가까운 값에서 12cm 이내에 있는 값들만 남겨 그 중앙값.
             '카메라에 가장 가까운 표면'만 골라내는 방식
"""

import numpy as np

# 심도를 읽을 때 관절 주변 몇 픽셀까지 볼지 (11x11 창)
PATCH_RADIUS = 5

# '앞면' 방법에서 같은 표면으로 볼 두께 (미터)
FRONT_BAND_M = 0.12


def _median(valid, scale):
    return float(np.median(valid)) * scale


def _nearest(valid, scale):
    return float(np.min(valid)) * scale


def _p10(valid, scale):
    return float(np.percentile(valid, 10)) * scale


def _front_surface(valid, scale):
    lowest = float(np.min(valid))
    band = valid[valid <= lowest + FRONT_BAND_M / scale]
    return float(np.median(band)) * scale


SAMPLERS = {
    "중앙값": _median,
    "최소": _nearest,
    "하위10%": _p10,
    "앞면": _front_surface,
}

# 2026-09-05 실측 검증 결과로 '하위10%'를 기본값으로 정했다.
# 팔꿈치를 고정한 채 몸만 돌리며 측정한 '방향에 따른 흔들림'(작을수록 좋음):
#   중앙값 30.2°  /  앞면 19.1°  /  최소 8.9°  /  하위10% 8.7°
# 중앙값은 손과 몸통 거리를 섞어버려 가장 나빴다.
# '앞면'은 12cm 창이 너무 넓어 뒤쪽 값이 섞였고, '최소'는 잡음 한 점에 흔들린다.
# 하위10%가 둘의 중간에서 가장 안정적이었다.
DEFAULT_SAMPLER = "하위10%"


def sample(patch, scale, method=DEFAULT_SAMPLER):
    """
    심도 patch(정수 배열)에서 거리(m)를 골라낸다.
    값이 0인 픽셀은 '측정 실패'라서 버린다. 쓸 값이 없으면 None.
    """
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return SAMPLERS[method](valid, scale)
