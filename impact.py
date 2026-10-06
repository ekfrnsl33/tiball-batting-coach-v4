"""
임팩트(타격 순간) 프레임 찾기 (명세서 6번)

명세서 6번 방식:
  배트·공을 따로 추적하는 장치가 없으므로, **손목이 가장 빠르게 움직인
  프레임**을 타격 순간으로 본다.

구현에서 신경 쓴 것:

1) 속도를 앞뒤 프레임 차이로 잰다 (중앙차분).
   바로 앞 프레임과의 차이만 쓰면 속도가 프레임 사이에 걸쳐 어긋난다.
   앞뒤를 함께 보면 그 프레임 자체의 속도가 된다.

2) 한 프레임 잡음으로 엉뚱한 곳이 뽑히지 않게 이동평균으로 다듬는다.

3) 화면 좌표(픽셀) 속도와 실제 3차원(미터) 속도를 둘 다 계산한다.
   각도는 3차원으로 재기로 정했지만(2026-09-05 검증), 빠르게 움직이는
   손목은 심도가 튀기 쉬워서 봉우리 찾기에는 픽셀 속도가 더 안정적일
   수 있다. 어느 쪽이 나은지 실제 스윙 데이터로 비교하려고 둘 다 남겼다.

4) 녹화 구간의 맨 끝에서 봉우리가 잡히면 스윙이 잘렸을 수 있으므로 경고한다.
"""

import numpy as np

from angles import point_from_record

# 이동평균 창 크기 (프레임). 60fps에서 5프레임 = 약 83ms
SMOOTH_WINDOW = 5

# 봉우리가 이 프레임 수 안쪽 끝에서 잡히면 스윙이 잘렸다고 본다
EDGE_MARGIN = 3

# 최고 봉우리의 이 비율 이상인 다른 봉우리가 있으면 '경쟁 봉우리'로 본다.
# 스윙 말고 다른 큰 움직임(배트를 다시 드는 동작, 걸어 나오기 등)이 섞였다는
# 신호이고, 그런 움직임이 조금만 더 빨랐으면 엉뚱한 프레임이 뽑혔다는 뜻이다.
COMPETING_RATIO = 0.8

# 같은 봉우리로 볼 최소 간격 (프레임). 60fps에서 15프레임 = 0.25초
MIN_PEAK_GAP = 15

# 손목 위치를 이만큼의 프레임으로 중앙값 필터를 걸어 한 프레임짜리 튐을 없앤다.
# 스윙이 가장 빠른 구간에서는 영상이 번져 MediaPipe가 손목을 한 프레임씩
# 엉뚱한 곳으로 잡는 일이 있는데, 그 튐이 그대로 '최고 속도'가 되어버린다.
# (2026-09-05 실측: 다듬지 않으면 손목 속도가 24.1 m/s로 나왔는데, 사람 손목이
#  낼 수 있는 속도를 넘는다. 3프레임 중앙값 필터로 20.7 m/s가 됐다.)
POSITION_MEDIAN_WINDOW = 3


def _moving_average(values, window):
    """빈 값(nan)을 건너뛰는 이동평균."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        chunk = values[lo:hi]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            out[i] = chunk.mean()
    return out


def _median_filter_positions(positions, window):
    """한 프레임짜리 위치 튐을 없앤다 (앞뒤를 함께 보고 중앙값을 취함)."""
    if window <= 1:
        return positions
    half = window // 2
    out = []
    for i in range(len(positions)):
        lo, hi = max(0, i - half), min(len(positions), i + half + 1)
        chunk = [p for p in positions[lo:hi] if p is not None]
        out.append(None if not chunk else np.median(np.vstack(chunk), axis=0))
    return out


def wrist_positions(records, wrist_index, camera, mode):
    """프레임별 손목 위치. 심도를 못 읽은 프레임은 None."""
    positions = []
    for record in records:
        point = point_from_record(record, wrist_index, camera, mode)
        positions.append(None if point is None else np.asarray(point,
                                                               dtype=float))
    return positions


def speed_series(positions, fps):
    """
    프레임별 손목 속도 (초당 이동량).
    앞뒤 프레임을 함께 보는 중앙차분을 쓴다.
    """
    n = len(positions)
    speeds = np.full(n, np.nan)
    for i in range(1, n - 1):
        before, after = positions[i - 1], positions[i + 1]
        if before is None or after is None:
            continue
        # 앞뒤 두 프레임 간격이므로 시간은 2/fps
        speeds[i] = float(np.linalg.norm(after - before)) * fps / 2.0
    return speeds


def find_peaks(smoothed, ratio=COMPETING_RATIO, gap=MIN_PEAK_GAP):
    """
    서로 떨어져 있는 큰 봉우리들을 찾는다 (높은 순으로 훑으면서,
    이미 뽑은 봉우리와 gap 프레임 안에 있으면 같은 봉우리로 보고 건너뛴다).
    반환: [(프레임 번호, 속도)] — 프레임 순서
    """
    values = np.asarray(smoothed, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    threshold = float(finite.max()) * ratio

    order = np.argsort(np.where(np.isnan(values), -np.inf, values))[::-1]
    picked = []
    for i in order:
        value = values[i]
        if not np.isfinite(value) or value < threshold:
            break
        if all(abs(int(i) - j) >= gap for j, _ in picked):
            picked.append((int(i), float(value)))
    return sorted(picked)


def find_impact(records, wrist_indices, camera, fps, mode="3D(m)"):
    """
    임팩트 프레임을 찾는다.

    wrist_indices: 손목 랜드마크 번호. 하나만 줘도 되고 여러 개를 줘도 된다.
      **양손을 함께 주는 것을 권한다.** 배트는 두 손으로 잡으므로 임팩트에서는
      두 손이 함께 빨라지는데, 한쪽이 몸에 가려 잘못 추적되면 그 손만으로는
      엉뚱한 프레임이 뽑힌다 (2026-09-05 실측: 뒷손만 쓰면 3.5초 지점의
      다른 동작을 임팩트로 골랐고, 양손 중 빠른 쪽을 쓰면 올바른 프레임을 골랐다).

    반환 dict:
      index        : 임팩트로 판단한 프레임 번호
      speed        : 다듬기 전 속도 배열
      smoothed     : 다듬은 속도 배열
      peak_speed   : 그 프레임의 속도
      near_edge    : 녹화 끝에 너무 가까운지 (스윙이 잘렸을 가능성)
      valid_ratio  : 속도를 계산할 수 있었던 프레임 비율
      peaks        : 크게 솟은 봉우리들 [(프레임, 속도)]
      competing    : 임팩트와 떨어져 있는 경쟁 봉우리들
    """
    if isinstance(wrist_indices, (int, np.integer)):
        wrist_indices = [int(wrist_indices)]

    per_wrist = []
    for index in wrist_indices:
        positions = wrist_positions(records, index, camera, mode)
        positions = _median_filter_positions(positions,
                                             POSITION_MEDIAN_WINDOW)
        per_wrist.append(speed_series(positions, fps))

    # 손이 여러 개면 '더 빠른 쪽'을 쓴다 (한쪽 추적 실패에 강하다).
    # np.fmax는 한쪽이 빈 값이면 다른 쪽을 쓰고, 둘 다 비면 빈 값을 준다.
    speeds = np.fmax.reduce(np.vstack(per_wrist), axis=0)
    smoothed = _moving_average(speeds, SMOOTH_WINDOW)

    if np.all(np.isnan(smoothed)):
        return None

    index = int(np.nanargmax(smoothed))
    n = len(records)
    peaks = find_peaks(smoothed)
    return {
        "index": index,
        "speed": speeds,
        "smoothed": smoothed,
        "peak_speed": float(smoothed[index]),
        "near_edge": index < EDGE_MARGIN or index >= n - EDGE_MARGIN,
        "valid_ratio": float(np.mean(~np.isnan(speeds))),
        "peaks": peaks,
        # 경쟁 봉우리가 있으면 '전체 최댓값' 규칙이 위태롭다는 신호
        "competing": [p for p in peaks if abs(p[0] - index) >= MIN_PEAK_GAP],
        "mode": mode,
    }
