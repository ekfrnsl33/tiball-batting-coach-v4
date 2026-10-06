"""
MediaPipe Pose 모듈 (앞으로 모든 단계에서 재사용)

MediaPipe = 구글이 만든 자세 인식 라이브러리. 영상 속 사람의 관절 위치
33개를 찾아준다. 우리는 학습을 따로 시키지 않고 사전학습 모델을 그대로 쓴다.
(명세서 1번: "33개 랜드마크, 사전학습 모델 그대로 사용")

주의: 이 맥에 설치된 mediapipe 0.10.35에는 예제에서 흔히 보이는
`mp.solutions.pose`(구형 API)가 없다. 신형 Tasks API를 써야 하고,
모델 파일(models/pose_landmarker_full.task)이 따로 필요하다.
골격선 그리기 도우미도 없어서 직접 그린다.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_full.task"

# ---------------------------------------------------------------------------
# 명세서 3번: 33개 중 실제로 쓰는 6개 지점 (좌/우 각각)
# ---------------------------------------------------------------------------
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# 좌타/우타에 따라 쓸 랜드마크 세트.
# 명세서 3번: 자동 인식 없이 촬영 준비 화면에서 연구자가 버튼으로 사전 선택.
LANDMARK_SETS = {
    "right": {  # 우타자 → 뒷팔/뒷다리는 오른쪽
        "shoulder": RIGHT_SHOULDER,
        "elbow": RIGHT_ELBOW,
        "wrist": RIGHT_WRIST,
        "hip": RIGHT_HIP,
        "knee": RIGHT_KNEE,
        "ankle": RIGHT_ANKLE,
    },
    "left": {   # 좌타자 → 뒷팔/뒷다리는 왼쪽
        "shoulder": LEFT_SHOULDER,
        "elbow": LEFT_ELBOW,
        "wrist": LEFT_WRIST,
        "hip": LEFT_HIP,
        "knee": LEFT_KNEE,
        "ankle": LEFT_ANKLE,
    },
}

LANDMARK_NAMES = {
    LEFT_SHOULDER: "왼어깨", RIGHT_SHOULDER: "오른어깨",
    LEFT_ELBOW: "왼팔꿈치", RIGHT_ELBOW: "오른팔꿈치",
    LEFT_WRIST: "왼손목", RIGHT_WRIST: "오른손목",
    LEFT_HIP: "왼골반", RIGHT_HIP: "오른골반",
    LEFT_KNEE: "왼무릎", RIGHT_KNEE: "오른무릎",
    LEFT_ANKLE: "왼발목", RIGHT_ANKLE: "오른발목",
}

# ---------------------------------------------------------------------------
# 명세서 3번 / 10번: γ = 가시성 임계값.
# 이 값보다 낮으면 "그 관절이 화면에 제대로 안 보인다"고 보고 버린다.
# ⚠️ 아직 확정된 값이 아님 — 파일럿 테스트 후 확정 예정.
# ---------------------------------------------------------------------------
VISIBILITY_THRESHOLD = 0.5

# 골격선을 그릴 때 어느 점끼리 이을지 (MediaPipe 표준 연결)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
]


class PoseExtractor:
    """녹화된 프레임에서 관절 33개 위치를 뽑아낸다."""

    def __init__(self, model_path=MODEL_PATH, min_confidence=0.5):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe 모델 파일이 없습니다: {model_path}")

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path)),
            # VIDEO 모드 = 연속된 프레임임을 알려주면 추적이 더 안정적이다
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def detect(self, color_bgr, timestamp_ms):
        """
        한 프레임에서 관절을 찾는다.
        반환: 랜드마크 33개 리스트. 각 항목은
              {"index", "x", "y", "visibility"} (x, y는 픽셀 좌표)
              사람을 못 찾으면 None.
        """
        # MediaPipe는 RGB 순서를 기대하지만 OpenCV/RealSense는 BGR 순서다
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO 모드는 시각이 항상 증가해야 한다
        ts = int(timestamp_ms)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self._landmarker.detect_for_video(mp_image, ts)
        if not result.pose_landmarks:
            return None

        h, w = color_bgr.shape[:2]
        landmarks = []
        for idx, lm in enumerate(result.pose_landmarks[0]):
            landmarks.append({
                "index": idx,
                # MediaPipe는 0~1 비율로 주므로 픽셀 좌표로 바꾼다
                "x": lm.x * w,
                "y": lm.y * h,
                "visibility": lm.visibility,
            })
        return landmarks


def is_visible(landmark, threshold=VISIBILITY_THRESHOLD):
    """명세서 3번의 가시성 필터."""
    return landmark["visibility"] >= threshold


def draw_skeleton(image, landmarks, highlight_indices=(),
                  threshold=VISIBILITY_THRESHOLD):
    """
    골격선을 그린 새 이미지를 만든다.
    highlight_indices에 든 지점(= 실제로 판정에 쓰는 6개)은 크고 진하게,
    나머지는 작고 흐리게 그린다.
    """
    canvas = image.copy()
    highlight = set(highlight_indices)

    # 먼저 선
    for a, b in POSE_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        la, lb = landmarks[a], landmarks[b]
        if not (is_visible(la, threshold) and is_visible(lb, threshold)):
            continue
        thick = 3 if (a in highlight and b in highlight) else 1
        color = (0, 255, 0) if thick == 3 else (160, 160, 160)
        cv2.line(canvas,
                 (int(la["x"]), int(la["y"])),
                 (int(lb["x"]), int(lb["y"])),
                 color, thick)

    # 그 위에 점
    for lm in landmarks:
        visible = is_visible(lm, threshold)
        if lm["index"] in highlight:
            radius, color = 6, (0, 128, 255) if visible else (0, 0, 255)
        else:
            radius, color = 2, (200, 200, 200) if visible else (80, 80, 80)
        cv2.circle(canvas, (int(lm["x"]), int(lm["y"])), radius, color, -1)

    return canvas


def skeleton_only(shape, landmarks, highlight_indices=(),
                  threshold=VISIBILITY_THRESHOLD, background=24):
    """
    원본 영상 없이 **골격선만** 그린 이미지를 만든다.

    명세서 9번: 브라우저(아동이 보는 화면)에는 골격선만 표시된 처리 이미지만
    노출하고, 원본 얼굴 등 식별 정보는 내보내지 않는다. 이 함수의 결과물이
    브라우저로 나가는 이미지다. 원본은 서버 안에만 둔다.
    """
    canvas = np.full((shape[0], shape[1], 3), background, dtype=np.uint8)
    return draw_skeleton(canvas, landmarks,
                         highlight_indices=highlight_indices,
                         threshold=threshold)
