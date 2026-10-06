"""
[진단용] RealSense가 어떤 조합으로 스트리밍 가능한지 한 번에 확인하는 스크립트.

test_camera.py는 "640x480@30 컬러+심도"라는 한 가지 조합만 시도한다.
이 스크립트는 카메라가 실제로 지원한다고 보고하는 모든 조합을 출력하고,
여러 조합을 차례로 켜보면서 어디까지 되는지 확인한다.

실행 (관리자 권한 필요):
  cd ~/Desktop/tiball_project
  sudo ./venv/bin/python diagnose_camera.py
"""

import sys

import pyrealsense2 as rs

TRY_CONFIGS = [
    # (설명, 폭, 높이, fps)
    ("640x480 @30", 640, 480, 30),
    ("640x480 @15", 640, 480, 15),
    ("480x270 @30", 480, 270, 30),
    ("424x240 @15", 424, 240, 15),
]


def line(title):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main():
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("[오류] RealSense를 찾지 못했습니다.")
        sys.exit(1)

    dev = devices[0]

    line("[1] 장치 정보")
    for label, key in [
        ("이름", rs.camera_info.name),
        ("시리얼", rs.camera_info.serial_number),
        ("펌웨어", rs.camera_info.firmware_version),
        ("USB 방식", rs.camera_info.usb_type_descriptor),
        ("제품 라인", rs.camera_info.product_line),
    ]:
        try:
            print(f"  {label:9s}: {dev.get_info(key)}")
        except Exception as e:
            print(f"  {label:9s}: (읽기 실패: {e})")

    # -----------------------------------------------------------------------
    # 카메라가 "지금 이 USB 연결에서" 지원한다고 보고하는 조합 목록.
    # USB 2.0으로 연결되면 이 목록 자체가 크게 줄어든다.
    # -----------------------------------------------------------------------
    line("[2] 지원 스트림 목록 (현재 USB 연결 기준)")
    for sensor in dev.query_sensors():
        sname = sensor.get_info(rs.camera_info.name)
        modes = set()
        for p in sensor.get_stream_profiles():
            if p.is_video_stream_profile():
                v = p.as_video_stream_profile()
                st = p.stream_type()
                if st in (rs.stream.depth, rs.stream.color):
                    modes.add((str(st).replace("stream.", ""),
                               v.width(), v.height(), p.fps(),
                               str(p.format()).replace("format.", "")))
        if not modes:
            continue
        print(f"  [{sname}]")
        for st, w, h, fps, fmt in sorted(modes):
            print(f"    {st:6s} {w:4d}x{h:<4d} {fps:3d}fps  {fmt}")

    # -----------------------------------------------------------------------
    # 실제로 켜보기. 컬러 단독 / 심도 단독 / 둘 다를 각각 시도한다.
    # -----------------------------------------------------------------------
    line("[3] 실제 스트리밍 시도")

    def attempt(desc, enable_fn):
        pipeline = rs.pipeline()
        config = rs.config()
        try:
            enable_fn(config)
            pipeline.start(config)
        except Exception as e:
            print(f"  {desc:32s} → 실패: {e}")
            return False
        try:
            frames = pipeline.wait_for_frames(5000)
            got = []
            if frames.get_color_frame():
                got.append("컬러")
            if frames.get_depth_frame():
                got.append("심도")
            print(f"  {desc:32s} → 성공 (수신: {', '.join(got) or '없음'})")
            return True
        except Exception as e:
            print(f"  {desc:32s} → 시작은 됐지만 프레임 없음: {e}")
            return False
        finally:
            pipeline.stop()

    for desc, w, h, fps in TRY_CONFIGS:
        attempt(
            f"컬러만 {desc}",
            lambda c, w=w, h=h, fps=fps: c.enable_stream(
                rs.stream.color, w, h, rs.format.bgr8, fps),
        )

    for desc, w, h, fps in TRY_CONFIGS:
        attempt(
            f"심도만 {desc}",
            lambda c, w=w, h=h, fps=fps: c.enable_stream(
                rs.stream.depth, w, h, rs.format.z16, fps),
        )

    for desc, w, h, fps in TRY_CONFIGS:
        def both(c, w=w, h=h, fps=fps):
            c.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            c.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        attempt(f"컬러+심도 {desc}", both)

    print()
    print("진단 끝. 위에서 '성공'이 뜬 조합 중 가장 높은 해상도/fps를 쓰면 됩니다.")


if __name__ == "__main__":
    main()
