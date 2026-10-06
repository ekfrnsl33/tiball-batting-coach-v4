"""
RealSense D435i 카메라 모듈 (앞으로 모든 단계에서 재사용)

맥에서 D435i를 쓰려면 까다로운 조건 세 가지가 다 맞아야 한다.
그 처리를 여기 한 곳에 모아두고, 다른 파일에서는 이렇게만 쓰면 된다.

    from camera import RealSenseCamera

    with RealSenseCamera() as cam:
        clip = cam.record(seconds=2.0)     # 2초 녹화
        for frame in clip:
            frame["color"]   # 컬러 이미지 (BGR, numpy)
            frame["depth"]   # 심도 이미지 (정수, numpy)
        cam.depth_to_meters(frame["depth"][y, x])   # 실제 거리(m)로 환산

맥에서 필요한 조건 세 가지:
  1) 관리자 권한(sudo)으로 실행해야 한다.
     일반 권한이면 카메라를 열 수조차 없다 (RS2_USB_STATUS_ACCESS).
  2) macOS의 UVCAssistant(시스템 카메라 드라이버)를 잠깐 종료시킨
     직후에 카메라를 잡아야 한다. 안 그러면 카메라는 열리지만
     영상 프레임이 하나도 안 들어온다.
     UVCAssistant는 맥이 자동으로 되살리는 서비스라 종료해도 안전하다.
  3) USB 3.x로 연결되어야 한다. USB 2.0 케이블이면 640x480@30을
     컬러+심도 동시에 보낼 수 없다.
"""

import subprocess
import time

import numpy as np
import pyrealsense2 as rs

import depth_sampling

# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
WIDTH = 640
HEIGHT = 480

# 60fps를 쓰는 이유: 명세서 6번은 "손목이 가장 빠른 프레임"을 임팩트로 보는데,
# 30fps면 프레임 간격이 33ms라 그 순간을 놓치기 쉽고 영상도 번진다.
# 60fps면 간격이 17ms로 절반이 되고 노출도 짧아져 번짐이 줄어든다.
# ⚠️ 실내가 어두우면 영상이 더 어두워질 수 있음 → 그럴 때는 30으로 되돌린다.
FPS = 60

WARMUP_FRAMES = 45       # 자동노출이 안정될 때까지 버리는 프레임 수 (약 0.75초)
FRAME_TIMEOUT_MS = 8000  # 프레임 하나를 기다리는 최대 시간
# 카메라 잡기를 시도할 때만 쓰는 짧은 대기시간.
# UVC 드라이버와의 경합에 지면 프레임이 아예 안 오는데, 그걸 8초씩 기다리면
# 재시도 10번에 80초가 걸린다. 실패는 빨리 알아채고 다시 시도하는 게 낫다.
OPEN_FRAME_TIMEOUT_MS = 1200
OPEN_MAX_ROUNDS = 25     # 카메라 잡기 재시도 횟수

# "No device connected"가 이만큼 연달아 나오면 재시도를 포기한다.
# 이 오류는 UVC 드라이버와의 경합이 아니라 **장치가 USB 버스에서 빠진**
# 상태다. 더 시도해도 소용없고, 케이블을 다시 꽂아야 한다.
# (D435i는 전원 상태 설정이 실패하면 스스로 리셋되며 사라지는 경우가 있다)
NO_DEVICE_GIVE_UP = 4
UVC_KILL_WAIT = 0.3      # UVC 드라이버 종료 후 카메라를 잡기까지의 대기(초)

# 맥에서 카메라를 붙잡고 있는 시스템 서비스들
UVC_SERVICES = ("UVCAssistant", "VDCAssistant", "cameracaptured")

# 심도값을 읽을 때 주변 몇 픽셀까지 함께 볼지 (7x7 창)
DEPTH_PATCH_RADIUS = 3


class CameraError(RuntimeError):
    """카메라를 열지 못했을 때 발생."""


def _release_uvc_driver():
    """맥의 UVC 카메라 드라이버를 잠깐 내린다 (자동으로 되살아남)."""
    for name in UVC_SERVICES:
        subprocess.run(["killall", name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class RealSenseCamera:
    """컬러 + 심도를 정렬해서 내주는 D435i 래퍼."""

    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS,
                 warmup_frames=WARMUP_FRAMES, verbose=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.verbose = verbose

        self._pipeline = None
        self._align = None
        self._color_sensor = None
        self.depth_scale = None    # 심도 정수값 1이 몇 미터인지
        self.intrinsics = None     # 렌즈 정보 (나중에 3D 좌표 환산에 필요)
        self.device_info = {}

    # -- 열기 / 닫기 --------------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _start_once(self):
        """카메라 잡기 1회 시도. 성공하면 True."""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height,
                             rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.bgr8, self.fps)
        try:
            profile = pipeline.start(config)
            # 프레임이 정말 흐르는지 한 장 받아서 확인한다.
            # (여기서 막히는 게 맥의 전형적인 증상이라 꼭 확인해야 한다)
            pipeline.wait_for_frames(OPEN_FRAME_TIMEOUT_MS)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            return None, str(exc)

        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()

        dev = profile.get_device()
        self._color_sensor = next(
            (sensor for sensor in dev.query_sensors()
             if sensor.get_info(rs.camera_info.name) == "RGB Camera"), None)
        for label, key in (("name", rs.camera_info.name),
                           ("serial", rs.camera_info.serial_number),
                           ("firmware", rs.camera_info.firmware_version),
                           ("usb", rs.camera_info.usb_type_descriptor)):
            try:
                self.device_info[label] = dev.get_info(key)
            except Exception:
                pass
        return profile, None

    def open(self):
        """UVC 드라이버를 내리고 카메라를 잡는다."""
        last_error = None
        missing_streak = 0
        for round_no in range(1, OPEN_MAX_ROUNDS + 1):
            _release_uvc_driver()
            time.sleep(UVC_KILL_WAIT)
            profile, error = self._start_once()
            if profile is not None:
                self._log(f"[카메라] 연결 성공 (시도 {round_no}회) — "
                          f"{self.device_info.get('name', '?')}, "
                          f"USB {self.device_info.get('usb', '?')}")
                break
            last_error = error
            # 실패는 한 줄로 짧게 (25번까지 시도하므로 길면 화면이 지저분해진다)
            self._log(f"[카메라] 시도 {round_no}회 실패 — 재시도 중 "
                      f"({str(error).splitlines()[0][:60]})")

            # 장치가 버스에서 사라진 경우는 빨리 포기하고 사용자에게 알린다
            if "No device connected" in str(error):
                missing_streak += 1
                if missing_streak >= NO_DEVICE_GIVE_UP:
                    raise CameraError(
                        "카메라가 USB에서 사라졌습니다 (No device connected). "
                        "소프트웨어 문제가 아니라 장치가 버스에서 빠진 상태입니다. "
                        "→ USB 케이블을 뽑았다가 다시 꽂고 프로그램을 다시 "
                        "실행해 주세요. 자주 반복되면 케이블이나 USB 허브를 "
                        "바꿔 보세요.")
            else:
                missing_streak = 0
            time.sleep(0.3)
        else:
            raise CameraError(
                "카메라를 열지 못했습니다. 확인할 것: "
                "(1) sudo로 실행했는지, (2) USB 3.0 이상으로 연결됐는지, "
                "(3) 케이블을 뽑았다 다시 꽂아볼 것. "
                f"마지막 오류: {last_error}")

        # 자동노출 안정화 — 이걸 건너뛰면 영상이 아주 어둡게 나온다
        for _ in range(self.warmup_frames):
            self._pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        self._log(f"[카메라] 자동노출 안정화 {self.warmup_frames}프레임 완료")
        return self

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            self._log("[카메라] 종료")

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # -- 프레임 받기 --------------------------------------------------------

    def read(self):
        """정렬된 컬러 + 심도 한 쌍을 numpy 배열로 돌려준다."""
        if self._pipeline is None:
            raise CameraError("카메라가 열려 있지 않습니다. open()을 먼저 호출하세요.")

        frames = self._pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        return {
            "color": np.asanyarray(color_frame.get_data()).copy(),
            "depth": np.asanyarray(depth_frame.get_data()).copy(),
            "timestamp_ms": frames.get_timestamp(),
        }

    def record(self, seconds=2.0):
        """
        명세서 2번의 '2~4초 단기 녹화'.
        실시간 스트리밍이 아니라, 짧게 녹화해두고 나중에 분석한다.
        """
        # 카메라에 미리 쌓여 있던 낡은 프레임을 먼저 버린다.
        # 안 그러면 60장을 순식간에 읽어버려서 실제로는 2초보다 짧은
        # 구간만 녹화된다.
        for _ in range(200):
            if not self._pipeline.poll_for_frames():
                break

        target = int(round(seconds * self.fps))
        clip = []
        for _ in range(target):
            frame = self.read()
            if frame is not None:
                clip.append(frame)
        self._log(f"[카메라] {seconds}초 녹화 완료 — {len(clip)}/{target} 프레임")
        return clip

    # -- 노출(셔터) 제어 -----------------------------------------------------
    #
    # 왜 필요한가: 스윙이 가장 빠른 구간에서 팔이 번져 찍히면 MediaPipe가
    # 손목·팔꿈치를 엉뚱한 곳으로 잡는다. 2026-09-05 실측에서 팔꿈치 각도가
    # 한 프레임에 99°나 튀었다. 노출 시간을 짧게 하면 번짐이 줄어든다.
    # 대가는 화면이 어두워지는 것이고, 감도(gain)를 올려 보정한다.
    #
    # D435i 컬러 센서의 노출 값은 100마이크로초 단위다.
    # (기본값 166 = 16.6ms = 60fps 한 프레임 길이와 일치)

    COLOR_EXPOSURE_UNIT_US = 100

    def color_option_range(self, option):
        """컬러 센서 설정값의 허용 범위를 알려준다. 지원 안 하면 None."""
        sensor = self._color_sensor
        if sensor is None or not sensor.supports(option):
            return None
        r = sensor.get_option_range(option)
        return {"min": r.min, "max": r.max, "step": r.step,
                "default": r.default}

    def set_auto_exposure(self, limit_ms=None):
        """
        자동노출을 켠다. limit_ms를 주면 **노출 시간의 상한**만 정한다.

        이게 장소가 바뀌어도 다시 맞추지 않아도 되는 방법이다.
        밝기는 카메라가 알아서 맞추되, 노출이 상한을 넘지 못하므로
        모션 블러가 항상 억제된다. 체육관이 거실보다 밝으면 자동으로
        더 짧은 노출을 골라 오히려 더 선명해진다.

        상한 기능을 지원하지 않는 펌웨어면 그냥 자동노출만 켜고 False를
        돌려준다 (그 경우 set_manual_exposure로 직접 고정해야 한다).
        """
        sensor = self._color_sensor
        if sensor is None:
            return False
        sensor.set_option(rs.option.enable_auto_exposure, 1)

        if limit_ms is None:
            self._log("[카메라] 컬러 자동노출 (상한 없음)")
            return True

        toggle = getattr(rs.option, "auto_exposure_limit_toggle", None)
        limit_option = getattr(rs.option, "auto_exposure_limit", None)
        if limit_option is None or not sensor.supports(limit_option):
            self._log("[카메라] ⚠ 이 펌웨어는 자동노출 상한을 지원하지 않습니다 "
                      "→ 자동노출만 켜짐")
            return False

        if toggle is not None and sensor.supports(toggle):
            sensor.set_option(toggle, 1)

        rng = self.color_option_range(limit_option)
        # 상한 값의 단위가 펌웨어마다 달라서 범위로 판별한다.
        # 최대값이 2만을 넘으면 마이크로초, 그보다 작으면 100마이크로초 단위.
        unit_us = 1.0 if (rng and rng["max"] > 20000) else 100.0
        value = limit_ms * 1000.0 / unit_us
        if rng:
            value = min(max(value, rng["min"]), rng["max"])
        sensor.set_option(limit_option, value)
        actual = value * unit_us / 1000.0
        self._log(f"[카메라] 컬러 자동노출 + 상한 {actual:.2f}ms "
                  f"(설정값 {value:.0f})")
        return True

    def get_color_exposure_ms(self):
        """카메라가 지금 실제로 쓰고 있는 노출 시간(ms). 자동노출 확인용."""
        sensor = self._color_sensor
        if sensor is None or not sensor.supports(rs.option.exposure):
            return None
        value = sensor.get_option(rs.option.exposure)
        return value * self.COLOR_EXPOSURE_UNIT_US / 1000.0

    def get_color_gain(self):
        sensor = self._color_sensor
        if sensor is None or not sensor.supports(rs.option.gain):
            return None
        return sensor.get_option(rs.option.gain)

    def set_power_line_frequency(self, hz=60):
        """
        형광등·LED 깜빡임(플리커) 대응. 한국은 60Hz.
        노출을 짧게 쓰면 조명 깜빡임이 프레임마다 밝기 차이로 나타날 수 있다.
        체육관 조명에서 특히 중요하다.
        """
        sensor = self._color_sensor
        option = getattr(rs.option, "power_line_frequency", None)
        if sensor is None or option is None or not sensor.supports(option):
            return False
        # 값 규약: 0=끔, 1=50Hz, 2=60Hz, 3=자동
        sensor.set_option(option, 2 if hz == 60 else 1)
        self._log(f"[카메라] 조명 깜빡임 대응 {hz}Hz")
        return True

    def set_manual_exposure(self, exposure_ms, gain=None):
        """
        노출 시간을 직접 정한다 (밀리초).
        gain을 주면 감도도 함께 올려 어두워지는 걸 보정한다.
        """
        sensor = self._color_sensor
        if sensor is None:
            return False

        value = exposure_ms * 1000.0 / self.COLOR_EXPOSURE_UNIT_US
        rng = self.color_option_range(rs.option.exposure)
        if rng:
            value = min(max(value, rng["min"]), rng["max"])

        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, value)
        actual = value * self.COLOR_EXPOSURE_UNIT_US / 1000.0
        message = f"[카메라] 노출 고정 {actual:.2f}ms (설정값 {value:.0f})"

        if gain is not None:
            grange = self.color_option_range(rs.option.gain)
            if grange:
                gain = min(max(gain, grange["min"]), grange["max"])
            sensor.set_option(rs.option.gain, gain)
            message += f", 감도 {gain:.0f}"
        self._log(message)
        return True

    # -- 심도 읽기 ----------------------------------------------------------

    def depth_to_meters(self, raw_value):
        """심도 정수값을 미터로 환산."""
        return float(raw_value) * self.depth_scale

    def depth_patch(self, depth_image, x, y, radius=DEPTH_PATCH_RADIUS):
        """
        (x, y) 주변의 심도 조각을 그대로 잘라서 준다 (거리 계산은 안 함).
        화면 밖이면 빈 배열.

        한 픽셀만 읽으면 안 되는 이유: 컬러와 심도를 정렬하면 사람 윤곽선
        주변에 '심도 그림자'(값 0 = 측정 실패)가 생긴다. 관절이 마침 윤곽에
        걸리면 0이 나오므로 주변을 함께 봐야 한다.
        """
        h, w = depth_image.shape[:2]
        x, y = int(round(x)), int(round(y))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return np.empty(0, dtype=depth_image.dtype)
        return depth_image[y0:y1, x0:x1]

    def sample_depth(self, depth_image, x, y, radius=DEPTH_PATCH_RADIUS,
                     method=None):
        """
        (x, y) 지점의 거리(m)를 읽는다. 값이 없으면 None.
        method로 읽는 방식을 고를 수 있다 (depth_sampling.SAMPLERS 참고).
        """
        patch = self.depth_patch(depth_image, x, y, radius)
        if patch.size == 0:
            return None
        return depth_sampling.sample(patch, self.depth_scale,
                                     method or depth_sampling.DEFAULT_SAMPLER)

    def pixel_to_point(self, x, y, depth_m):
        """
        픽셀 좌표 + 거리 → 실제 3차원 좌표(X, Y, Z, 단위 미터).

        각도를 정확히 재려면 필요하다. 픽셀 단위(x, y)와 미터 단위(거리)를
        그냥 섞으면 각도가 왜곡되기 때문에, 렌즈 정보로 같은 미터 단위로
        환산해준다.
        """
        return rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [float(x), float(y)], float(depth_m))
