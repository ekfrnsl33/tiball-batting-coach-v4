"""
[촬영 준비] 실시간 미리보기 — 카메라 각도와 설 자리를 눈으로 맞춘다

화면에 표시되는 것:
  - 실시간 영상 + 골격선
  - 초록 점선 = 안전 영역. 판정에 쓰는 관절이 이 안에 들어와야 안심.
  - 화면 테두리 색 = 초록(준비 완료) / 노랑(조정 필요)  ← 멀리서도 보임
  - 아래 패널 = 관절별 상태와 거리

조작 (창을 한 번 클릭해서 선택한 뒤 눌러야 먹습니다):
  q 또는 esc : 종료
  s          : 지금 화면 저장 (test_output/preview_snapshot.png)
  l          : 좌타/우타 전환

실행 (관리자 권한 필요):
  cd ~/Desktop/tiball_project
  sudo ./venv/bin/python preview.py
"""

import sys
import time
from pathlib import Path

import cv2

from camera import RealSenseCamera, CameraError
from pose import PoseExtractor, LANDMARK_SETS
from preview_ui import WINDOW, render

OUTPUT_DIR = Path(__file__).parent / "test_output"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    side = "right"

    try:
        cam = RealSenseCamera().open()
    except CameraError as exc:
        print(f"[오류] {exc}")
        sys.exit(1)

    print()
    print("미리보기 창이 열립니다. 창을 보면서 카메라 각도와 설 위치를 맞춰주세요.")
    print("  화면 테두리가 초록색이 되면 준비된 상태입니다.")
    print("  창을 한 번 클릭한 뒤:  q 종료 / s 저장 / l 좌타·우타 전환")
    print()

    last = time.monotonic()
    fps = 0.0

    try:
        with PoseExtractor() as extractor:
            while True:
                frame = cam.read()
                if frame is None:
                    continue

                landmarks = extractor.detect(frame["color"],
                                             frame["timestamp_ms"])
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now

                image, _, _, _ = render(
                    frame, landmarks, LANDMARK_SETS[side], cam, side, fps,
                    footer="q 종료  s 저장  l 좌우전환")
                cv2.imshow(WINDOW, image)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('s'):
                    path = OUTPUT_DIR / "preview_snapshot.png"
                    cv2.imwrite(str(path), image)
                    print(f"  저장: {path}")
                if key == ord('l'):
                    side = "left" if side == "right" else "right"
                    print(f"  전환: {'좌타자' if side == 'left' else '우타자'}")
    finally:
        cv2.destroyAllWindows()
        cam.close()


if __name__ == "__main__":
    main()
