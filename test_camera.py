"""
[1단계] RealSense 카메라 연결 테스트

목적:
  MediaPipe도 Flask도 아직 붙이지 않고, "카메라에서 RGB 영상과 심도(Depth) 영상이
  둘 다 잘 들어오는지"만 확인한다.

확인하는 것:
  1) RealSense D435i가 파이썬에서 열리는지
  2) RGB(컬러) 프레임이 들어오는지
  3) 심도(Depth) 프레임이 들어오는지
  4) 두 영상의 픽셀이 서로 맞춰지는지 (= Align, 명세서 2번의 "RGB-심도 정렬")
  5) 화면 중앙까지의 실제 거리가 미터 단위로 측정되는지

실행:
  cd ~/Desktop/tiball_project
  ./venv/bin/python test_camera.py
"""

import sys
from pathlib import Path

import numpy as np

# pyrealsense2 = 인텔 RealSense 카메라를 파이썬에서 다루는 공식 라이브러리.
# (카메라 켜기/끄기, 프레임 받아오기, 컬러-심도 정렬 등을 담당)
try:
    import pyrealsense2 as rs
except ImportError:
    print("[오류] pyrealsense2를 불러올 수 없습니다.")
    print("       이 라이브러리는 맥(Apple Silicon)용 설치 파일이 제공되지 않아서,")
    print("       소스 코드에서 직접 빌드해야 합니다. 빌드 단계를 먼저 진행해주세요.")
    sys.exit(1)

# OpenCV = 영상 처리 라이브러리. 여기서는 확인용 이미지 저장에만 쓴다.
import cv2

# ---------------------------------------------------------------------------
# 설정값 (나중에 숫자만 바꾸면 되도록 위쪽에 모아둠)
# ---------------------------------------------------------------------------
WIDTH = 640          # 가로 해상도
HEIGHT = 480         # 세로 해상도
FPS = 30             # 초당 프레임 수
WARMUP_FRAMES = 30   # 자동노출이 안정될 때까지 버리는 프레임 수
TEST_FRAMES = 10     # 실제로 확인할 프레임 수

OUTPUT_DIR = Path(__file__).parent / "test_output"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # 1) 카메라 연결 확인
    #    파이프라인(pipeline)은 "카메라에서 프레임이 흘러나오는 통로"라고 생각하면 된다.
    # -----------------------------------------------------------------------
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("[오류] 연결된 RealSense 카메라를 찾지 못했습니다.")
        print("       USB 케이블이 꽂혀 있는지, USB 3.0 포트인지 확인해주세요.")
        sys.exit(1)

    dev = devices[0]
    print("=" * 60)
    print("[1] 카메라 인식")
    print("=" * 60)
    print("  이름     :", dev.get_info(rs.camera_info.name))
    print("  시리얼   :", dev.get_info(rs.camera_info.serial_number))
    print("  펌웨어   :", dev.get_info(rs.camera_info.firmware_version))
    print("  USB 방식 :", dev.get_info(rs.camera_info.usb_type_descriptor))

    # -----------------------------------------------------------------------
    # 2) 컬러 + 심도 두 스트림을 동시에 켠다
    # -----------------------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    print()
    print("=" * 60)
    print("[2] 스트림 시작")
    print("=" * 60)
    profile = pipeline.start(config)

    # 심도 스케일 = 심도 영상의 정수값 1이 실제 몇 미터인지를 나타내는 환산 계수.
    # D435i는 보통 0.001 (즉 1 = 1mm)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"  심도 스케일 : {depth_scale}  (정수값 1 = {depth_scale * 1000:.3f} mm)")

    # align = 컬러 영상과 심도 영상의 픽셀 위치를 서로 맞춰주는 도구.
    # 두 렌즈가 물리적으로 조금 떨어져 있어서, 이걸 안 하면
    # "컬러의 (x,y)"와 "심도의 (x,y)"가 같은 지점을 가리키지 않는다.
    # 명세서 2번에서 요구하는 부분이 바로 이것.
    align = rs.align(rs.stream.color)

    try:
        # 자동노출 안정화용으로 앞쪽 프레임은 그냥 버린다
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()

        print()
        print("=" * 60)
        print("[3] 프레임 수신 확인")
        print("=" * 60)

        color_image = None
        depth_frame_last = None
        ok_count = 0

        for i in range(TEST_FRAMES):
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)   # 여기서 정렬이 일어남

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                print(f"  프레임 {i + 1:2d}: 수신 실패")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_frame_last = depth_frame

            # 화면 정중앙 픽셀까지의 거리를 미터로 읽어본다
            cx, cy = WIDTH // 2, HEIGHT // 2
            center_m = depth_frame.get_distance(cx, cy)

            # 심도값이 0인 픽셀 = 측정 실패(너무 가깝거나, 반사가 안 되거나)
            valid_ratio = float((depth_image > 0).mean())

            print(f"  프레임 {i + 1:2d}: "
                  f"컬러 {color_image.shape}, 심도 {depth_image.shape}, "
                  f"중앙거리 {center_m:.3f} m, 유효심도 {valid_ratio * 100:.1f}%")
            ok_count += 1

        # -------------------------------------------------------------------
        # 3) 눈으로 확인할 수 있게 이미지 2장 저장
        # -------------------------------------------------------------------
        if color_image is not None and depth_frame_last is not None:
            depth_image = np.asanyarray(depth_frame_last.get_data())

            # 심도 영상은 원래 사람 눈에 안 보이는 숫자 배열이라,
            # 가까울수록 파랑 / 멀수록 빨강처럼 색을 입혀서 저장한다.
            depth_colored = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            cv2.imwrite(str(OUTPUT_DIR / "test_color.png"), color_image)
            cv2.imwrite(str(OUTPUT_DIR / "test_depth.png"), depth_colored)

            # 정렬이 제대로 됐는지 보려면 두 장을 반투명하게 겹쳐보는 게 가장 확실하다.
            overlay = cv2.addWeighted(color_image, 0.6, depth_colored, 0.4, 0)
            cv2.imwrite(str(OUTPUT_DIR / "test_aligned_overlay.png"), overlay)

            print()
            print("=" * 60)
            print("[4] 확인용 이미지 저장 완료")
            print("=" * 60)
            print("  ", OUTPUT_DIR / "test_color.png", " (컬러 원본)")
            print("  ", OUTPUT_DIR / "test_depth.png", " (심도, 색 입힘)")
            print("  ", OUTPUT_DIR / "test_aligned_overlay.png", " (겹쳐보기 - 정렬 확인용)")

        print()
        if ok_count == TEST_FRAMES:
            print(f"[성공] {TEST_FRAMES}프레임 모두 컬러 + 심도가 정상 수신됩니다.")
        else:
            print(f"[주의] {TEST_FRAMES}프레임 중 {ok_count}프레임만 수신됐습니다.")
            print("       USB 포트를 바꿔보거나 해상도/FPS를 낮춰볼 필요가 있습니다.")

    finally:
        # 어떤 오류가 나도 카메라는 반드시 꺼준다 (안 끄면 다음 실행 때 안 열림)
        pipeline.stop()
        print("\n[정리] 카메라를 정상적으로 종료했습니다.")


if __name__ == "__main__":
    main()
