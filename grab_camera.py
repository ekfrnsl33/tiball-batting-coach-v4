"""
[마지막 시도] macOS UVC 드라이버를 잠깐 내리고 그 틈에 RealSense를 잡는다.

배경:
  맥에는 UVCAssistant라는 시스템 카메라 드라이버가 있고, 이게 RealSense의
  영상 데이터 통로를 계속 붙잡고 있어서 librealsense로 영상이 안 들어온다.
  (제어 명령은 되는데 프레임만 안 오는 상태)

방법:
  UVCAssistant를 잠깐 종료시키고, 맥이 그걸 다시 켜기 전에
  librealsense가 먼저 카메라를 잡도록 여러 번 빠르게 시도한다.

안전성:
  UVCAssistant는 macOS가 자동으로 다시 실행하는 시스템 서비스다.
  종료해도 몇 초 뒤 스스로 되살아나고, 맥을 재시작하면 확실히 원상복구된다.
  (그 사이 잠깐 다른 앱에서 카메라가 안 보일 수 있다)

실행 (관리자 권한 필요):
  cd ~/Desktop/tiball_project
  sudo ./venv/bin/python grab_camera.py
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
import cv2

WIDTH, HEIGHT, FPS = 640, 480, 30
FRAME_TIMEOUT_MS = 8000
MAX_ROUNDS = 12          # 몇 번까지 재시도할지
OUTPUT_DIR = Path(__file__).parent / "test_output"


def kill_uvc():
    """맥의 UVC 카메라 드라이버를 잠깐 내린다 (자동으로 되살아남)."""
    for name in ("UVCAssistant", "VDCAssistant", "cameracaptured"):
        subprocess.run(["killall", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def try_stream(round_no):
    """카메라를 새로 열어서 심도+컬러 프레임을 받아본다. 성공하면 프레임 반환."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"  [{round_no:2d}] 시작 실패: {e}")
        return None

    try:
        # 자동노출 안정화 겸, 프레임이 실제로 흐르는지 확인
        align = rs.align(rs.stream.color)
        for _ in range(5):
            frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            print(f"  [{round_no:2d}] 프레임 일부 없음")
            return None

        color = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        center_m = depth_frame.get_distance(WIDTH // 2, HEIGHT // 2)
        print(f"  [{round_no:2d}] ★ 성공! 컬러 {color.shape}, 심도 {depth.shape}, "
              f"중앙거리 {center_m:.3f} m")
        return color, depth, center_m
    except Exception as e:
        print(f"  [{round_no:2d}] 프레임 대기 실패: {e}")
        return None
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rs.log_to_console(rs.log_severity.none)

    print("=" * 64)
    print("UVC 드라이버를 내리고 RealSense를 잡아봅니다")
    print("=" * 64)

    result = None
    for round_no in range(1, MAX_ROUNDS + 1):
        kill_uvc()
        # 맥이 드라이버를 다시 올리기 전의 짧은 틈을 노린다
        time.sleep(0.3)
        result = try_stream(round_no)
        if result:
            break
        time.sleep(1.0)

    if not result:
        print()
        print("[실패] UVC 드라이버를 내려도 영상이 들어오지 않습니다.")
        print("       맥에서 심도 카메라를 직접 쓰는 건 여기서 막힙니다.")
        sys.exit(2)

    color, depth, center_m = result
    depth_colored = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(color, 0.6, depth_colored, 0.4, 0)

    cv2.imwrite(str(OUTPUT_DIR / "test_color.png"), color)
    cv2.imwrite(str(OUTPUT_DIR / "test_depth.png"), depth_colored)
    cv2.imwrite(str(OUTPUT_DIR / "test_aligned_overlay.png"), overlay)

    print()
    print("[성공] 확인용 이미지 3장을 test_output/ 에 저장했습니다.")
    print(f"       화면 중앙까지의 거리: {center_m:.3f} m")


if __name__ == "__main__":
    main()
