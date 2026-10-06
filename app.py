"""
[5단계] Flask 웹앱 — 명세서 8번의 3화면

  ① 촬영 준비  /          학생 선택, 좌타/우타, 프레이밍 미리보기, 촬영 시작
  ② 촬영 진행  /capture   카운트다운, 스윙 진행 상태
  ③ 결과 확인  /result    자세 이미지, 관절별 판정, 각도, 레벨·최근 이력

구조:
  카메라는 sudo로만 열리고 여는 데 시간이 걸리므로(맥 UVC 드라이버 문제),
  **작업 스레드 하나가 카메라를 계속 붙잡고** 있는다. 웹 화면은 그 스레드의
  상태를 조회하고 촬영을 요청할 뿐이다.
  MediaPipe는 스레드 안전하지 않으므로 이 스레드 안에서만 쓴다.

개인정보 (명세서 9번 — 2026-09-06 사용자 결정으로 변경됨):
  화면에는 **학생 모습(원본 영상) + 골격선**을 보여준다. 아이가 자기
  자세를 알아보려면 몸이 보이는 게 낫다는 판단이다.

  대신 원본이 밖으로 나가지 않게 두 가지를 지킨다.
    1) 웹 서버는 이 컴퓨터 자신(127.0.0.1)에만 열린다. 다른 기기에서는
       접속되지 않으므로 원본이 네트워크를 타지 않는다.
    2) 명세서 8번의 '같은 네트워크의 태블릿에서 결과 공유'를 나중에 켜면
       (BIND_HOST를 바꾸면) 원본 표시가 자동으로 꺼지고 골격선만 나간다.
  구글 시트로는 어떤 경우에도 영상·이미지를 보내지 않는다 (수치만).
  보관 기간과 삭제 시점은 명세서·논문에 명시하기로 했다.

실행 (관리자 권한 필요):
  cd ~/Desktop/tiball_project
  sudo ./venv/bin/python app.py
  → 브라우저에서 http://127.0.0.1:5050 열기
"""

import random
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import (Flask, Response, abort, jsonify, redirect,
                   render_template, request, send_file, session, url_for)

import impact as impact_mod
import storage
from angles import angle_fit_at, point_from_record
from camera import RealSenseCamera, CameraError
from capture import SIDE_LABEL, build_records
from judging import (GOOD, JOINT_LABEL, LEVEL_MULTIPLIER, MAX_LEVEL, PERFECT,
                     PROMOTE_AFTER, apply_grade, direction_phrases, feedback,
                     judge_joint, judge_session, score_for, tolerances)
from pose import (LANDMARK_SETS, LANDMARK_NAMES, PoseExtractor,
                  VISIBILITY_THRESHOLD, draw_skeleton, skeleton_only)
from preview_ui import MARGIN_RATIO, check_framing

APP_DIR = Path(__file__).parent
OUTPUT_DIR = APP_DIR / "test_output"

# 명세서 9번 관련 설정 (2026-09-06 결정)
SHOW_ORIGINAL = True          # 화면에 학생 모습(원본)을 함께 보여줄지
BIND_HOST = "127.0.0.1"       # 이 컴퓨터에서만 접속 가능

RECORD_SECONDS = 3.0
COUNTDOWN_SECONDS = 8
SWINGS_PER_SESSION = 3      # 4단계 결정: 3회 중앙값으로 1회 판정

# v4 요청 1번: 세션 1회차 촬영 전에만 재생하는 사전 음성 안내.
# 실제 재생은 브라우저(TTS)에서 하고, 서버는 그 발화가 끝났다는 신호
# (/api/briefing-done)를 기다렸다가 기존 8초 카운트다운을 시작한다.
# 신호가 오지 않는 경우(음성 미지원 등)를 대비한 안전장치로 최대 대기시간을 둔다.
BRIEFING_TEXT = ("티볼 타격 교정 프로그램에 대해 안내드리겠습니다. "
                 "준비 시간은 8초, 타격 촬영은 3초 동안 진행됩니다. "
                 "타격 이후에는 정확한 판정을 위해 1초 정도 멈춰주세요. "
                 "그럼 준비가 되셨으면 시작합니다.")
BRIEFING_TIMEOUT_SECONDS = 20

JOINTS = {
    "elbow": ("shoulder", "elbow", "wrist"),
    "knee": ("hip", "knee", "ankle"),
}

GREEN = (60, 200, 60)
YELLOW = (40, 200, 240)
RED = (60, 60, 235)

# v4 요청 3번: 사이드 내비게이션 4개 메뉴에 아이콘 추가(이모지)
NAV_ITEMS = [
    ("posture_guide", "🏏 티볼 자세"),
    ("prepare", "📷 촬영 준비"),
    ("result", "✅ 결과 확인"),
    ("interpretation_guide", "❓ 결과 해석 방법"),
]


# ---------------------------------------------------------------------------
# v3 요청 1번: 스윙별 개별 판정 + 점수
# ---------------------------------------------------------------------------

def build_swing_detail(per_swing, level):
    """
    관절별 스윙 1~3회차 각각의 각도·등급·점수·방향성 지시어를 만든다.
    per_swing: {"elbow": [각도,...], "knee": [각도,...]} — 실패한 스윙은 빠져 있다.
    반환: {"elbow": [{"angle","grade","score","phrase"}, ...], "knee": [...]}
    """
    detail = {}
    for joint, values in per_swing.items():
        rows = []
        for value in values:
            info = judge_joint(joint, value, level)
            phrase = None
            if info["grade"] != PERFECT:
                phrase = random.choice(
                    direction_phrases(joint, value, info["theta"], info["beta"]))
            rows.append({
                "angle": round(value, 1),
                "grade": info["grade"],
                "score": score_for(info["grade"]),
                "phrase": phrase,
            })
        detail[joint] = rows
    return detail


def total_score_of(swing_detail):
    return sum(row["score"] for rows in swing_detail.values() for row in rows)


def session_phrase(joint, grade, angle, theta, beta):
    """세션 등급(중앙값 기준)에 대한 방향성 지시어 하나. 완벽이면 None."""
    if grade == PERFECT:
        return None
    return random.choice(direction_phrases(joint, angle, theta, beta))


def build_baseline(results, baseline_row):
    """
    v3 요청 2번: 이번 세션 값을 그 학생의 최초 세션과 비교한다.
    baseline_row가 None이면(그 학생의 첫 기록이면) None을 돌려준다.
    """
    if not baseline_row:
        return None
    info = []
    for r in results:
        base_val = baseline_row.get(f"{r['joint']}_angle")
        if not base_val:
            continue
        base_val = float(base_val)
        delta = r["angle"] - base_val
        closer = abs(r["angle"] - r["theta"]) < abs(base_val - r["theta"])
        if delta > 0:
            direction = "펴짐"
        elif delta < 0:
            direction = "굽혀짐"
        else:
            direction = "변화 없음"
        info.append({"label": r["label"], "delta": round(abs(delta), 1),
                     "direction": direction, "closer": closer})
    return info or None


def build_result_from_history(student, timestamp=None):
    """
    v3 요청 3번: '마지막 결과 다시 보기' + '최근 판정 이력에서 회차 선택'.
    timestamp를 주면 그 학생의 그 시각 세션을, 안 주면 가장 최근 세션을
    CSV 기록으로 결과 화면과 같은 모양의 데이터로 다시 만들어준다
    (이미지는 저장된 파일을 그대로 씀).
    """
    row = (storage.session_at(student, timestamp) if timestamp
          else storage.latest_session(student))
    if not row:
        return None

    level = int(row.get("level_before") or 1)
    per_swing = {}
    results = []
    for joint in JOINTS:
        angle_text = row.get(f"{joint}_angle")
        if not angle_text:
            continue
        angle = float(angle_text)
        info = judge_joint(joint, angle, level)
        unc_text = row.get(f"{joint}_uncertainty")
        values = []
        for i in range(1, 4):
            v = row.get(f"{joint}_swing{i}")
            if v not in (None, ""):
                values.append(float(v))
        per_swing[joint] = values
        results.append({**info, "angle": round(angle, 1),
                        "phrase": session_phrase(joint, info["grade"], angle,
                                                 info["theta"], info["beta"]),
                        "uncertainty": float(unc_text) if unc_text else None,
                        "swings": [round(v, 1) for v in values]})

    swing_detail = build_swing_detail(per_swing, level)
    total_text = row.get("session_total_score")
    total_score = (int(float(total_text)) if total_text
                  else total_score_of(swing_detail))

    first_row = storage.first_session(student)
    is_first = first_row and first_row.get("timestamp") == row.get("timestamp")
    baseline = None if is_first else build_baseline(results, first_row)

    return {
        "student": student, "side": row.get("batting_side"),
        "grade": row.get("session_grade"),
        "feedback": feedback(row.get("session_grade"), results),
        "results": results,
        "event": None,
        "swings_used": int(row.get("swings") or 0),
        "image_path": row.get("image_shown") or row.get("image_skeleton"),
        "timestamp": row.get("timestamp"),
        "tee_height_cm": row.get("tee_height_cm") or "",
        "total_score": total_score,
        "swing_detail": swing_detail,
        "baseline": baseline,
        "from_history": True,
    }


# ---------------------------------------------------------------------------
# 브라우저로 내보낼 이미지 만들기 (골격선만)
# ---------------------------------------------------------------------------

def show_original():
    """
    원본을 보여줘도 되는 상황인지 판단한다.
    이 컴퓨터에만 열려 있을 때만 허용한다. 네트워크에 열면 원본이
    네트워크를 타므로 자동으로 골격선만 내보낸다.
    """
    return SHOW_ORIGINAL and BIND_HOST in ("127.0.0.1", "localhost")


def frame_image(frame, landmarks, joint_set):
    """
    화면에 보여줄 바탕 이미지.
    허용되면 학생 모습 + 골격선, 아니면 골격선만.
    """
    shape = frame["color"].shape
    highlight = sorted(set(joint_set.values()))
    if landmarks is None:
        if show_original():
            return frame["color"].copy()
        return np.full((shape[0], shape[1], 3), 24, dtype=np.uint8)
    if show_original():
        return draw_skeleton(frame["color"], landmarks,
                             highlight_indices=highlight)
    return skeleton_only(shape, landmarks, highlight_indices=highlight)


def web_image(frame, landmarks, joint_set, border_color):
    """미리보기 이미지 — 바탕 위에 안전 영역 점선과 상태 테두리를 그린다."""
    canvas = frame_image(frame, landmarks, joint_set)
    h, w = canvas.shape[:2]
    mx, my = int(w * MARGIN_RATIO), int(h * MARGIN_RATIO)
    for x in range(mx, w - mx, 20):
        cv2.line(canvas, (x, my), (min(x + 10, w - mx), my), (70, 120, 70), 1)
        cv2.line(canvas, (x, h - my), (min(x + 10, w - mx), h - my),
                 (70, 120, 70), 1)
    for y in range(my, h - my, 20):
        cv2.line(canvas, (mx, y), (mx, min(y + 10, h - my)), (70, 120, 70), 1)
        cv2.line(canvas, (w - mx, y), (w - mx, min(y + 10, h - my)),
                 (70, 120, 70), 1)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), border_color, 10)
    return canvas


def to_jpeg(image):
    ok, buffer = cv2.imencode(".jpg", image,
                              [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buffer.tobytes() if ok else b""


# ---------------------------------------------------------------------------
# 카메라 작업 스레드
# ---------------------------------------------------------------------------

class CameraWorker(threading.Thread):
    """카메라를 붙잡고 있으면서 미리보기와 촬영을 담당한다."""

    daemon = True

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._briefing_ack = threading.Event()
        self._request = None          # 촬영 요청 {student, side}

        self.phase = "starting"       # starting/idle/countdown/recording/analyzing/error
        self.message = "카메라를 준비하는 중입니다..."
        self.side = "left"
        self.countdown = None
        self.take = 0
        self.framing = {"ok": False, "advice": None, "rows": [], "hip": None}
        self.preview = b""
        self.result = None
        self.camera_info = {}

        # v2 CSV 로깅 확장 (2026-09-14): 촬영 환경 메모.
        # 다음 세션 요청이 올 때까지 유지되는 값이 아니라, 세션 1회당
        # request_session()에서 받아 _run_session에 그대로 넘긴다.

    # -- 외부에서 부르는 것 -------------------------------------------------

    def snapshot(self):
        with self._lock:
            return {
                "phase": self.phase,
                "message": self.message,
                "side": self.side,
                "countdown": self.countdown,
                "take": self.take,
                "swings_total": SWINGS_PER_SESSION,
                "framing": self.framing,
                "has_result": self.result is not None,
            }

    def preview_jpeg(self):
        with self._lock:
            return self.preview

    def request_session(self, student, side, env_note="", tee_height_cm=""):
        with self._lock:
            if self.phase != "idle":
                return False
            self._request = {"student": student, "side": side,
                             "env_note": env_note,
                             "tee_height_cm": tee_height_cm}
            self.side = side
            return True

    def set_side(self, side):
        with self._lock:
            self.side = side

    def acknowledge_briefing(self):
        """
        v4 요청 1번: 브라우저가 사전 안내 음성 재생을 끝냈다고 알려올 때 호출.
        1회차 촬영 전, 이 신호를 받아야(또는 안전시간 초과 시) 카운트다운을
        시작한다.
        """
        self._briefing_ack.set()
        return True

    def cancel(self):
        """
        진행 중인 촬영을 중단한다.
        중간에 아이가 준비가 안 됐거나 자세가 흐트러졌을 때 쓴다.
        """
        with self._lock:
            self._request = None
        self._cancel.set()
        return True

    def stop(self):
        self._stop.set()

    # -- 내부 --------------------------------------------------------------

    def _set(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def run(self):
        try:
            cam = RealSenseCamera().open()
        except CameraError as exc:
            self._set(phase="error", message=str(exc))
            return

        self._set(camera_info=cam.device_info)
        cam.set_power_line_frequency(60)
        extractor = PoseExtractor()
        self._set(phase="idle", message="준비되었습니다.")

        try:
            while not self._stop.is_set():
                with self._lock:
                    pending = self._request
                    self._request = None
                if pending:
                    self._run_session(cam, extractor, pending)
                else:
                    self._tick(cam, extractor)
        finally:
            extractor.close()
            cam.close()

    def _tick(self, cam, extractor, state=None):
        """한 프레임 읽어 미리보기와 프레이밍 상태를 갱신한다."""
        frame = cam.read()
        if frame is None:
            return None, None
        with self._lock:
            side = self.side
        joint_set = LANDMARK_SETS[side]
        landmarks = extractor.detect(frame["color"], frame["timestamp_ms"])

        if landmarks is None:
            rows, ok, advice, hip = [], False, "사람이 안 보입니다", None
        else:
            rows, ok, advice = check_framing(landmarks, joint_set,
                                             frame["depth"], cam,
                                             frame["color"].shape)
            hip = next((r["depth"] for r in rows if r["role"] == "hip"), None)

        color = RED if state == "recording" else (GREEN if ok else YELLOW)
        image = web_image(frame, landmarks, joint_set, color)
        self._set(
            preview=to_jpeg(image),
            framing={
                "ok": bool(ok),
                "advice": advice,
                "hip": round(hip, 2) if hip else None,
                "rows": [{"name": r["name"],
                          "state": ("화면밖" if not r["inside"]
                                    else "심도없음" if r["depth"] is None
                                    else "가려짐" if not r["visible"]
                                    else "정상"),
                          "depth": round(r["depth"], 2) if r["depth"] else None}
                         for r in rows],
            })
        return frame, landmarks

    def _aborted(self):
        """취소 버튼이 눌렸는지 확인한다."""
        if self._cancel.is_set():
            self._cancel.clear()
            self._set(phase="idle", countdown=None, take=0,
                      message="촬영을 취소했습니다. 다시 시작할 수 있습니다.")
            return True
        return False

    def _run_session(self, cam, extractor, job):
        student, side = job["student"], job["side"]
        env_note = job.get("env_note", "")
        tee_height_cm = job.get("tee_height_cm", "")
        joint_set = LANDMARK_SETS[side]
        swings = []
        self._cancel.clear()

        for take in range(1, SWINGS_PER_SESSION + 1):
            if self._aborted():
                return

            # v4 요청 1번: 1회차 촬영 전에만 사전 음성 안내를 재생하고, 그
            # 발화가 끝날 때까지 기다린 뒤에 기존 카운트다운을 시작한다.
            if take == 1:
                self._briefing_ack.clear()
                self._set(phase="briefing", take=take,
                          message="사전 안내 음성을 재생합니다")
                briefing_end = time.monotonic() + BRIEFING_TIMEOUT_SECONDS
                while not self._briefing_ack.is_set():
                    if self._aborted():
                        return
                    if time.monotonic() >= briefing_end:
                        break  # 음성 미지원 등으로 신호가 안 오면 그냥 진행
                    self._tick(cam, extractor)

            self._set(phase="countdown", take=take,
                      message=f"{take}번째 스윙 준비")
            end = time.monotonic() + COUNTDOWN_SECONDS
            while True:
                if self._aborted():
                    return
                remain = end - time.monotonic()
                if remain <= 0:
                    break
                self._set(countdown=int(np.ceil(remain)))
                self._tick(cam, extractor)

            self._set(phase="recording", countdown=None,
                      message=f"{take}번째 스윙 촬영 중")
            self._tick(cam, extractor, state="recording")
            clip = cam.record(seconds=RECORD_SECONDS)

            self._set(phase="analyzing", message="분석 중")
            records = build_records(cam, extractor, clip)
            if len(records) >= 10:
                analyzed = self._analyze(records, cam, joint_set)
                if analyzed:
                    swings.append(analyzed)

        self._finish(student, side, joint_set, swings, env_note, tee_height_cm)

    def _analyze(self, records, cam, joint_set):
        wrists = [LANDMARK_SETS["left"]["wrist"],
                  LANDMARK_SETS["right"]["wrist"]]
        found = impact_mod.find_impact(records, wrists, cam, cam.fps, "3D(m)")
        if found is None:
            return None
        peak = found["index"]
        out = {"impact": peak, "competing": len(found["competing"]),
               "joints": {}, "coords": {}, "record": records[peak]}
        for joint, roles in JOINTS.items():
            indices = [joint_set[role] for role in roles]
            out["joints"][joint] = angle_fit_at(records, indices, cam, peak)
            out["coords"][joint] = point_from_record(records[peak],
                                                     indices[1], cam)
        return out

    def _finish(self, student, side, joint_set, swings, env_note="",
               tee_height_cm=""):
        if not swings:
            self._set(phase="idle", countdown=None,
                      message="스윙을 찾지 못했습니다. 다시 촬영해 주세요.")
            return

        angles, uncertainty, spreads, per_swing = {}, {}, {}, {}
        for joint in JOINTS:
            values = [s["joints"][joint]["angle"] for s in swings
                      if s["joints"][joint]
                      and s["joints"][joint]["angle"] is not None]
            per_swing[joint] = [round(v, 1) for v in values]
            if not values:
                angles[joint] = None
                continue
            angles[joint] = float(np.median(values))
            if len(values) >= 2:
                sd = float(np.std(values, ddof=1))
                spreads[joint] = sd
                uncertainty[joint] = sd / np.sqrt(len(values))

        before = storage.get_state(student)
        results, grade = judge_session(angles, before["level"])
        after, event = apply_grade(dict(before), grade)
        # 2026-09-17: '좋음' 누적은 승급·강등 판단(apply_grade, 손대지 않음)과
        # 무관한 표시 전용 값이라 여기서 따로만 센다.
        after["good_count"] = before.get("good_count", 0) + (1 if grade == GOOD else 0)
        storage.set_state(student, after)

        # v3 요청 1번: 스윙별 개별 판정·점수 (기존 3회 중앙값 판정/레벨업은 위에서
        # 이미 끝났고, 여기부터는 순전히 "추가로 보여줄 정보"만 만든다).
        swing_detail = build_swing_detail(per_swing, before["level"])
        total_score = total_score_of(swing_detail)
        # v3 요청 2번: 개인별 베이스라인(그 학생의 최초 기록)은 이번 세션을
        # 기록하기 전에 미리 읽어 둬야 "최초 세션 그 자체"와 비교하지 않는다.
        baseline = build_baseline(results, storage.first_session(student))

        best = swings[len(swings) // 2]
        row = {
            "timestamp": storage.now_text(),
            "student_id": student, "batting_side": side,
            "level_before": before["level"], "level_after": after["level"],
            "perfect_count": after["perfect_count"],
            "retry_count": after["retry_count"], "event": event or "",
            "swings": len(swings), "session_grade": grade,
            "impact_frame": best["impact"],
            "joint_coords": storage.format_coords(best["coords"]),
        }
        for r in results:
            row[f"{r['joint']}_angle"] = round(r["angle"], 2)
            row[f"{r['joint']}_grade"] = r["grade"]
            unc = uncertainty.get(r["joint"])
            row[f"{r['joint']}_uncertainty"] = round(unc, 2) if unc else ""

        # ↓ 2026-09-14 CSV 로깅 확장 (v2 전용)
        # 임팩트 프레임 시점, 판정에 쓴 6개 지점의 MediaPipe 가시성 원값.
        # (좌타/우타에 따라 joint_set이 이미 좌/우 중 쓰는 쪽만 담고 있다)
        best_landmarks = best["record"]["landmarks"]
        for role, idx in joint_set.items():
            row[f"{role}_visibility"] = round(best_landmarks[idx]["visibility"], 3)
        row["gamma_threshold"] = VISIBILITY_THRESHOLD

        # 3회 스윙 각각의 원본 각도(그림4의 "81.4° / 83.5° / 80.5°"와 동일 값).
        # 성공한 스윙 수가 3보다 적으면 남는 칸은 빈 값으로 둔다.
        for joint in JOINTS:
            values = per_swing.get(joint, [])
            for i in range(3):
                row[f"{joint}_swing{i + 1}"] = values[i] if i < len(values) else ""

        row["env_note"] = env_note

        # ↓ 2026-09-16 v3 통합본: 스윙별 개별 점수 + 티 높이
        row["tee_height_cm"] = tee_height_cm
        for joint in JOINTS:
            rows_detail = swing_detail.get(joint, [])
            for i in range(3):
                row[f"{joint}_swing{i + 1}_score"] = (
                    rows_detail[i]["score"] if i < len(rows_detail) else "")
        row["session_total_score"] = total_score

        # 결과 이미지 두 장을 학생별 폴더에 남긴다 (2026-09-06 결정).
        #   *_shown.png    학생 모습 + 골격선 — 논문에는 얼굴을 가려서 사용
        #   *_skeleton.png 골격선만        — 가릴 필요 없이 바로 사용 가능
        # 보관·삭제 방침은 data/results/보관안내.txt 에 함께 적어둔다.
        record = best["record"]
        highlight = sorted(set(joint_set.values()))
        shown_image = frame_image(record, record["landmarks"], joint_set)
        skeleton_image = skeleton_only(record["color"].shape,
                                       record["landmarks"],
                                       highlight_indices=highlight)

        shown_path, skeleton_path = storage.result_paths(student,
                                                        row["timestamp"])
        cv2.imwrite(str(shown_path), shown_image)
        cv2.imwrite(str(skeleton_path), skeleton_image)
        # sudo로 저장하면 root 소유가 되어 나중에 파일을 옮기거나 논문에
        # 넣으려 할 때 권한 오류가 난다
        storage.fix_permissions(shown_path)
        storage.fix_permissions(skeleton_path)
        row["image_shown"] = str(shown_path.relative_to(APP_DIR))
        row["image_skeleton"] = str(skeleton_path.relative_to(APP_DIR))
        browser_image = shown_image

        # 이미지 경로까지 채운 뒤에 기록을 남긴다.
        # 기록은 이 컴퓨터 안에만 저장한다 (2026-09-06 결정: 클라우드 연동 없음).
        storage.append_session(row)

        self._set(
            phase="idle", countdown=None, take=0,
            message="분석이 끝났습니다.",
            result={
                "student": student, "side": side,
                "grade": grade, "feedback": feedback(grade, results),
                "results": [{
                    "joint": r["joint"], "label": r["label"],
                    "angle": round(r["angle"], 1), "theta": r["theta"],
                    "error": round(r["error"], 1),
                    "alpha": r["alpha"], "beta": r["beta"],
                    "grade": r["grade"],
                    "phrase": session_phrase(r["joint"], r["grade"],
                                             r["angle"], r["theta"], r["beta"]),
                    "uncertainty": (round(uncertainty[r["joint"]], 1)
                                    if r["joint"] in uncertainty else None),
                    "spread": (round(spreads[r["joint"]], 1)
                               if r["joint"] in spreads else None),
                    "swings": per_swing[r["joint"]],
                } for r in results],
                "before": before, "after": after, "event": event,
                "swings_used": len(swings),
                "image": to_jpeg(browser_image),
                "image_path": None,
                "timestamp": row["timestamp"],
                "tee_height_cm": tee_height_cm,
                "total_score": total_score,
                "swing_detail": swing_detail,
                "baseline": baseline,
                "from_history": False,
            })


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

app = Flask(__name__)
# 로컬(127.0.0.1) 전용 1인 앱이라 값 자체는 민감하지 않다. '마지막으로 본
# 학생'을 세션 쿠키에 기억해 두는 용도로만 쓴다 (v3 요청 3번).
app.secret_key = "tiball-v3-local-only"
worker = CameraWorker()


@app.context_processor
def inject_nav():
    return {"nav_items": NAV_ITEMS}


@app.route("/")
def prepare():
    students = storage.load_students()
    # 결과 화면에서 '같은 학생 다시 촬영'으로 왔을 때 그 학생을 미리 골라둔다
    selected = request.args.get("student")
    if selected not in students:
        selected = session.get("last_student")
    if selected not in students:
        selected = next(iter(students), None)
    side = request.args.get("side")
    if side in SIDE_LABEL:
        worker.set_side(side)
    else:
        side = worker.snapshot()["side"]
    session["last_student"] = selected
    return render_template("prepare.html",
                           students=students,
                           selected_student=selected,
                           selected_side=side,
                           status=worker.snapshot(),
                           swings=SWINGS_PER_SESSION,
                           side_labels=SIDE_LABEL,
                           active_menu="prepare")


@app.route("/capture", methods=["POST", "GET"])
def capture():
    if request.method == "POST":
        student = request.form.get("student", "S1")
        side = request.form.get("side", "left")
        env_note = request.form.get("env_note", "").strip()
        tee_height_cm = request.form.get("tee_height_cm", "").strip()
        session["last_student"] = student
        if not worker.request_session(student, side, env_note, tee_height_cm):
            return redirect(url_for("prepare"))
        return redirect(url_for("capture"))
    return render_template("capture.html", status=worker.snapshot(),
                           swings=SWINGS_PER_SESSION, active_menu="prepare",
                           briefing_text=BRIEFING_TEXT)


@app.route("/result")
def result():
    # v3 요청 3번: '학생별로 전부 보기' + '마지막 결과 다시 보기' +
    # '최근 판정 이력에서 회차 선택'.
    # 화면 위쪽 학생 선택으로 아무 학생이나 고를 수 있고, ?at=시각 이 있으면
    # (이력 표에서 클릭한 경우) 그 학생의 그 회차를 최우선으로 보여준다.
    # 둘 다 없으면 방금 이 컴퓨터에서 촬영한 결과를 쓰고, 그것도 없거나
    # 다른 학생을 보고 있으면 그 학생의 최근 CSV 기록으로 다시 만들어준다.
    students = storage.load_students()
    if not students:
        return redirect(url_for("prepare"))

    student_param = request.args.get("student") or session.get("last_student")
    if student_param not in students:
        student_param = next(iter(students))
    session["last_student"] = student_param

    at_param = request.args.get("at")
    if at_param:
        data = build_result_from_history(student_param, at_param)
    else:
        live = worker.result
        if live and live.get("student") == student_param:
            data = live
        else:
            data = build_result_from_history(student_param)

    state = storage.get_state(student_param)
    remaining = max(0, PROMOTE_AFTER - state["perfect_count"])
    return render_template(
        "result.html", data=data, state=state,
        students=students, selected_student=student_param,
        recent=storage.recent_sessions(student_param, limit=5),
        multiplier=LEVEL_MULTIPLIER[state["level"]],
        remaining=remaining, max_level=MAX_LEVEL,
        tolerance_rows=[(JOINT_LABEL[j], *tolerances(j, state["level"]))
                        for j in JOINTS],
        active_menu="result")


@app.route("/posture-guide")
def posture_guide():
    """v3 요청 3-1번: '티볼 자세' 안내 화면. 기준각 숫자는 절대 넣지 않는다."""
    return render_template("posture_guide.html", active_menu="posture_guide")


@app.route("/interpretation-guide")
def interpretation_guide():
    """v3 요청 3-4번: '결과 해석 방법'. 점수제만 설명하고 레벨업은 언급하지 않는다."""
    return render_template("interpretation_guide.html",
                           active_menu="interpretation_guide")


@app.route("/media/<path:relpath>")
def media(relpath):
    """
    CSV에 저장된 결과 이미지를 다시 보여줄 때 쓴다 ('마지막 결과 다시 보기').
    data/results/ 바깥은 절대 내보내지 않는다 (경로 조작 방지).
    """
    full = (APP_DIR / relpath).resolve()
    results_root = storage.RESULTS_DIR.resolve()
    if results_root not in full.parents and full != results_root:
        abort(403)
    if not full.is_file():
        abort(404)
    return send_file(full)


@app.route("/preview.jpg")
def preview_jpg():
    data = worker.preview_jpeg()
    if not data:
        return Response(status=204)
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/result.jpg")
def result_jpg():
    data = worker.result
    if not data:
        return Response(status=204)
    return Response(data["image"], mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/status")
def api_status():
    return jsonify(worker.snapshot())


@app.route("/api/students", methods=["POST"])
def api_add_student():
    """학생 추가 (명세서 8번의 고정 6명에서 늘어날 수 있게)."""
    payload = request.get_json(silent=True) or {}
    added, error = storage.add_student(payload.get("student_id", ""))
    if error:
        return jsonify({"ok": False, "error": error}), 400
    return jsonify({"ok": True, "student_id": added})


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    worker.cancel()
    return jsonify({"ok": True})


@app.route("/api/briefing-done", methods=["POST"])
def api_briefing_done():
    """v4 요청 1번: 브라우저의 사전 안내 음성 재생이 끝났다는 신호."""
    worker.acknowledge_briefing()
    return jsonify({"ok": True})


@app.route("/api/side", methods=["POST"])
def api_side():
    side = request.json.get("side", "left")
    worker.set_side(side)
    return jsonify({"side": side})


@app.route("/admin/delete-all", methods=["GET", "POST"])
def admin_delete_all():
    """
    개발·테스트 단계에서 쌓인 기록을 한 번에 지우는 숨겨진 관리자 화면.
    평상시 화면 어디에도 이 경로로 가는 링크는 없다 — 주소를 직접 입력해야
    들어올 수 있다. 실수로 지우지 않도록 확인 문구를 정확히 입력해야
    삭제가 실행된다 (2단계 확인).

    2026-09-14: v1과 v2 양쪽에 동일하게 추가됨. 각 버전은 storage.py의
    SESSIONS_CSV가 가리키는 자신의 파일만 지운다.
    """
    CONFIRM_WORD = "삭제합니다"
    if request.method == "POST":
        if request.form.get("confirm_text", "") != CONFIRM_WORD:
            return render_template("admin_delete.html", done=False,
                                   error=f'확인 문구를 정확히 "{CONFIRM_WORD}"라고 입력해야 삭제됩니다.',
                                   csv_name=storage.SESSIONS_CSV.name,
                                   confirm_word=CONFIRM_WORD)
        summary = storage.delete_all_records()
        return render_template("admin_delete.html", done=True,
                               error=None, summary=summary,
                               csv_name=storage.SESSIONS_CSV.name,
                               confirm_word=CONFIRM_WORD)
    return render_template("admin_delete.html", done=False, error=None,
                           csv_name=storage.SESSIONS_CSV.name,
                           confirm_word=CONFIRM_WORD)


def port_in_use(host, port):
    """이미 같은 프로그램이 떠 있는지 확인한다."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return False
        except OSError:
            return True


def main():
    # 카메라를 건드리기 전에 확인한다. 이미 떠 있는데 또 켜면 두 프로그램이
    # 카메라를 서로 빼앗아 촬영이 중간에 멈춘다.
    if port_in_use(BIND_HOST, 5050):
        print()
        print("=" * 70)
        print("  이미 이 프로그램이 실행 중입니다 (포트 5050 사용 중)")
        print("=" * 70)
        print("  먼저 실행 중인 창에서 Ctrl+C 로 종료한 뒤 다시 실행하세요.")
        print("  창을 못 찾겠으면 아래 명령으로 정리할 수 있습니다:")
        print("      sudo pkill -f 'python -u app.py'")
        print("=" * 70)
        print()
        return

    worker.start()
    print()
    print("=" * 70)
    print("  티볼 타격 자세 교정 프로그램")
    print("=" * 70)
    print("  브라우저에서 다음 주소를 열어주세요:")
    print("      http://127.0.0.1:5050")
    print()
    if show_original():
        print("  화면 표시: 학생 모습 + 골격선 (이 컴퓨터에서만 접속 가능)")
    else:
        print("  화면 표시: 골격선만 (네트워크에 열려 있어 원본은 보내지 않음)")
    print("  카메라를 잡는 데 10~20초 걸릴 수 있습니다.")
    print("  종료하려면 이 창에서 Ctrl+C 를 누르세요.")
    print("=" * 70)
    print()
    try:
        # use_reloader=False: 켜면 프로세스가 두 번 떠서 카메라를 두 번 잡으려 한다
        app.run(host=BIND_HOST, port=5050, threaded=True,
                use_reloader=False)
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
