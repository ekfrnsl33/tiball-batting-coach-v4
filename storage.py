"""
학생별 상태 저장 + 세션 기록 (명세서 7번, 9번)

두 가지를 다룬다.

1) 학생별 레벨 상태 — data/students.json
   명세서 7번: 참여자가 6명이고, 레벨과 누적 카운트는 프로그램 전체가 아니라
   **학생 개인별로** 저장돼야 한다. 프로그램을 껐다 켜거나 다른 아이가
   이어서 써도 각자의 진행 상태가 유지되어야 한다.

2) 세션 기록 — data/sessions.csv
   명세서 9번: 매 세션마다 관절 좌표값, 각도, 판정 등급, 촬영 일시,
   학생ID, 레벨 상태를 남긴다.
   저장 우선순위는 **로컬 먼저**다. 구글 시트 동기화는 맨 마지막에 붙이며,
   그게 실패해도 판정 기능은 영향받지 않아야 한다.

개인정보 (명세서 9번 / 논문 Ⅲ-3):
  실명은 저장하지 않는다. 연구용 코드(학생ID)만 쓴다.
  원본 영상은 이 파일들에 들어가지 않는다.
"""

import csv
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from judging import MIN_LEVEL

DATA_DIR = Path(__file__).parent / "data"
STUDENTS_FILE = DATA_DIR / "students.json"
# v4에서는 v1(sessions.csv)·v2(sessions_v2.csv)·v3(sessions_v3.csv)를 건드리지
# 않도록 별도 파일을 쓴다. (2026-09-18 v4: 3차 전문가 검토 반영 작업)
SESSIONS_CSV = DATA_DIR / "sessions_v4.csv"

# 명세서 8번: 처음 목록은 6명. 화면에는 실명 대신 연구용 코드로 표시.
# 전학생 등으로 인원이 늘 수 있으므로 화면에서 추가할 수 있게 했다
# (add_student). 목록은 data/students.json 에 저장되고, 한번 추가한 학생은
# 계속 유지된다.
DEFAULT_STUDENT_IDS = ["S1", "S2", "S3", "S4", "S5", "S6"]

# 연구용 코드로 쓸 수 있는 형태.
# 이 값이 **폴더 이름으로 쓰이기 때문에** 제한이 필요하다.
# (`../` 같은 경로 문자가 들어가면 엉뚱한 곳에 폴더가 만들어진다)
STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9가-힣_-]{1,16}$")

SESSION_FIELDS = [
    "timestamp",         # 촬영 일시
    "student_id",        # 연구용 코드 (실명 아님)
    "batting_side",      # 좌타/우타
    "level_before",
    "level_after",
    "perfect_count",
    "retry_count",
    "event",             # 승급 / 강등 / (없음)
    "swings",            # 이 판정에 사용한 스윙 횟수
    "elbow_angle",
    "elbow_grade",
    "elbow_uncertainty",
    "knee_angle",
    "knee_grade",
    "knee_uncertainty",
    "session_grade",
    "impact_frame",
    "joint_coords",      # 관절 좌표값 (명세서 9번)
    "image_shown",       # 학생 모습 + 골격선 이미지 파일
    "image_skeleton",    # 골격선만 그린 이미지 파일

    # ↓ 2026-09-14 CSV 로깅 확장 (v2 전용, 기존 컬럼 뒤에 추가)
    #   불확실성이 크게 나오는 원인을 사후 진단하기 위한 항목들.
    "gamma_threshold",    # 가시성 임계값(γ). pose.py VISIBILITY_THRESHOLD
    "shoulder_visibility",  # 임팩트 프레임 시점, 판정에 쓴 6개 지점의
    "elbow_visibility",      # MediaPipe 가시성(visibility) 원값.
    "wrist_visibility",      # 좌우 중 실제 사용된 쪽만 기록한다.
    "hip_visibility",
    "knee_visibility",
    "ankle_visibility",
    "elbow_swing1", "elbow_swing2", "elbow_swing3",  # 스윙 1/2/3회차
    "knee_swing1", "knee_swing2", "knee_swing3",     # 개별 산출 각도
    "env_note",           # 촬영 환경 메모 (선택 입력)

    # ↓ 2026-09-16 v3 통합본: 스윙별 개별 점수제 + 티 높이 (기존 컬럼 뒤에 추가)
    "tee_height_cm",           # 티 높이(cm), 선택 입력, 비우면 빈 값
    "elbow_swing1_score", "elbow_swing2_score", "elbow_swing3_score",
    "knee_swing1_score", "knee_swing2_score", "knee_swing3_score",
    "session_total_score",     # 두 관절 × 3스윙 점수 합 (다시연습만 60 ~ 완벽만 300)
]

RESULTS_DIR = DATA_DIR / "results"

# 이미지가 쌓이는 폴더라 보관·삭제 방침을 파일로 함께 남긴다.
RETENTION_NOTE = """[결과 이미지 보관 안내]

이 폴더에는 학생별(S1~S6)로 세션마다 두 장의 이미지가 저장됩니다.

  *_shown.png     학생 모습 + 골격선  ← 얼굴이 포함됩니다
  *_skeleton.png  골격선만            ← 얼굴이 포함되지 않습니다

파일 이름의 숫자는 촬영 일시입니다 (예: 20260906_003937).
같은 일시가 data/sessions_v4.csv 의 timestamp 열과 짝을 이룹니다.

보관 및 처리:
  - 이 컴퓨터 안에만 저장되며 네트워크나 클라우드로 전송되지 않습니다.
  - 논문에 실을 때는 얼굴을 가리거나 *_skeleton.png 를 사용합니다.
  - 연구 종료 후 이 폴더 전체를 삭제합니다.
"""


def _blank_state():
    # good_count(좋음 누적)는 2026-09-17 v3 추가: 승급·강등 판단에는 전혀
    # 쓰이지 않는 표시 전용 값이다. 판정 로직(judging.py)은 건드리지 않고
    # 여기 저장 계층에서만 따로 센다.
    return {"level": MIN_LEVEL, "perfect_count": 0, "retry_count": 0,
           "good_count": 0}


def _fix_permissions(path):
    """
    sudo로 실행하면 파일이 root 소유가 되어, 나중에 선생님이 엑셀로 CSV를
    열거나 파일을 옮기려 할 때 권한 오류가 난다. 실제 사용자에게 소유권을
    돌려주고 읽기 권한을 열어둔다.
    """
    try:
        os.chmod(path, 0o644 if path.is_file() else 0o755)
    except OSError:
        pass
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid:
        try:
            os.chown(path, int(uid), int(gid))
        except OSError:
            pass


# 다른 모듈에서도 쓸 수 있게 공개 이름을 하나 둔다
fix_permissions = _fix_permissions


def _write_atomic(path, text):
    """쓰다가 중단돼도 파일이 깨지지 않게 임시 파일에 쓴 뒤 바꿔치기한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp, path)
        _fix_permissions(path.parent)
        _fix_permissions(path)
    except Exception:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def load_students():
    """학생별 상태를 불러온다. 파일이 없으면 기본 6명으로 만든다."""
    if STUDENTS_FILE.exists():
        try:
            with open(STUDENTS_FILE, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            # 권한 문제나 파일 손상으로 못 읽으면 기본값으로 계속 간다
            # (판정은 되게 하고, 상태 저장만 실패했음을 알린다)
            print(f"[저장] 학생 상태를 읽지 못했습니다: {exc}")
            return {student_id: _blank_state()
                    for student_id in DEFAULT_STUDENT_IDS}
        # 목록에 없는 학생이 나중에 추가돼도 깨지지 않게 빈 상태로 채운다
        for student_id in DEFAULT_STUDENT_IDS:
            data.setdefault(student_id, _blank_state())
        # good_count 추가 전에 저장된 학생 파일에는 이 키가 없을 수 있다
        for state in data.values():
            state.setdefault("good_count", 0)
        return data

    data = {student_id: _blank_state()
            for student_id in DEFAULT_STUDENT_IDS}
    save_students(data)
    return data


def save_students(data):
    _write_atomic(STUDENTS_FILE,
                  json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def get_state(student_id):
    return load_students().get(student_id, _blank_state())


def set_state(student_id, state):
    data = load_students()
    data[student_id] = {
        "level": int(state["level"]),
        "perfect_count": int(state["perfect_count"]),
        "retry_count": int(state["retry_count"]),
        "good_count": int(state.get("good_count", 0)),
    }
    save_students(data)
    return data[student_id]


def _migrate_csv_header():
    """
    기록 항목이 늘어났을 때, 기존 CSV를 새 제목줄로 옮겨 적는다.
    안 하면 새 줄의 값이 제목과 어긋나게 들어간다.
    """
    if not SESSIONS_CSV.exists():
        return
    with open(SESSIONS_CSV, encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == SESSION_FIELDS:
            return
        rows = list(reader)
    with open(SESSIONS_CSV, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SESSION_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for old in rows:
            writer.writerow({key: old.get(key, "") for key in SESSION_FIELDS})
    print(f"[저장] 기록 항목이 늘어나 {SESSIONS_CSV.name} 제목줄을 갱신했습니다.")


def result_paths(student_id, timestamp):
    """
    그 학생의 결과 이미지 두 장을 저장할 경로를 만든다 (폴더도 만든다).
    반환: (얼굴 포함 이미지 경로, 골격선만 이미지 경로)
    """
    stamp = (timestamp.replace("-", "").replace(":", "")
             .replace(" ", "_"))
    folder = RESULTS_DIR / student_id
    folder.mkdir(parents=True, exist_ok=True)
    note = RESULTS_DIR / "보관안내.txt"
    if not note.exists():
        note.write_text(RETENTION_NOTE, encoding="utf-8")
        _fix_permissions(note)
    for path in (DATA_DIR, RESULTS_DIR, folder):
        _fix_permissions(path)
    return (folder / f"{stamp}_shown.png",
            folder / f"{stamp}_skeleton.png")


def add_student(student_id):
    """
    학생을 목록에 추가한다 (전학생 등).
    반환: (추가된 코드, 오류 메시지) — 성공하면 오류가 None.
    """
    student_id = (student_id or "").strip()
    if not student_id:
        return None, "연구용 코드를 입력해 주세요."
    if not STUDENT_ID_PATTERN.match(student_id):
        return None, ("연구용 코드는 영문·숫자·한글·- ·_ 만 쓸 수 있고 "
                      "1~16자여야 합니다. (폴더 이름으로도 쓰입니다)")
    data = load_students()
    if student_id in data:
        return None, f"'{student_id}'는 이미 목록에 있습니다."
    data[student_id] = _blank_state()
    save_students(data)
    return student_id, None


def append_session(row):
    """
    세션 기록 한 줄을 CSV에 덧붙인다.
    파일이 없으면 제목줄부터 만든다.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_csv_header()
    exists = SESSIONS_CSV.exists()
    with open(SESSIONS_CSV, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SESSION_FIELDS,
                                extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    _fix_permissions(DATA_DIR)
    _fix_permissions(SESSIONS_CSV)
    return SESSIONS_CSV


def recent_sessions(student_id, limit=5):
    """
    그 학생의 최근 판정 이력 (명세서 8번: 결과 화면 사이드 패널에 표시).
    """
    if not SESSIONS_CSV.exists():
        return []
    with open(SESSIONS_CSV, encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get("student_id") == student_id]
    return rows[-limit:][::-1]


def first_session(student_id):
    """
    그 학생의 가장 오래된(=최초) 세션 기록 한 줄.
    v3 요청 2번의 '개인별 베이스라인 대비 변화'에 쓴다. 기록이 없으면 None.
    """
    if not SESSIONS_CSV.exists():
        return None
    with open(SESSIONS_CSV, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("student_id") == student_id:
                return row
    return None


def session_at(student_id, timestamp):
    """
    그 학생의 특정 시각(timestamp) 세션 기록 한 줄.
    '최근 판정 이력'에서 원하는 회차를 콕 집어 다시 볼 때 쓴다.
    """
    if not SESSIONS_CSV.exists():
        return None
    with open(SESSIONS_CSV, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row.get("student_id") == student_id
                    and row.get("timestamp") == timestamp):
                return row
    return None


def latest_session(student_id):
    """
    그 학생의 가장 최근 세션 기록 한 줄.
    v3 요청 3번의 '마지막 결과 다시 보기'(다른 메뉴에서 돌아와도 그 학생의
    최근 결과를 다시 불러오기)에 쓴다. 기록이 없으면 None.
    """
    rows = recent_sessions(student_id, limit=1)
    return rows[0] if rows else None


def format_coords(coords):
    """
    관절 좌표를 CSV 한 칸에 넣을 문자열로 만든다.
    예: elbow=(0.12,-0.34,2.51); knee=(...)
    """
    parts = []
    for name, point in coords.items():
        if point is None:
            parts.append(f"{name}=None")
        else:
            x, y, z = point
            parts.append(f"{name}=({x:.3f},{y:.3f},{z:.3f})")
    return "; ".join(parts)


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 2026-09-14: 전체 기록 삭제 (개발·테스트 단계 정리용)
#
# ⚠️ 개발 검증 과정에서 쌓인 테스트 기록을 정리하기 위한 기능이다.
#    실제 연구에 참여하는 아동의 데이터에는 사용하지 않는다.
#    평상시 화면에는 노출되지 않고 숨겨진 관리자 경로(/admin/delete-all)
#    에서만 접근할 수 있다. app.py 쪽에서 2단계 확인을 거친 뒤에만 부른다.
#
# 지우는 것:
#   1) 세션 기록 CSV (sessions.csv 또는 sessions_v2.csv, 버전별로 자신의 것만)
#   2) 결과 이미지 전부 (data/results/ 안의 모든 학생 폴더, 보관안내.txt는 남김)
#   3) 학생별 레벨·누적횟수 — 목록(학생 코드)은 유지한 채 전부 1단계로 초기화
#      (CSV는 지워지는데 레벨만 남으면 앞뒤가 안 맞기 때문)
# ---------------------------------------------------------------------------

def delete_all_records():
    """
    이 버전(v1/v2/v3/v4, SESSIONS_CSV가 가리키는 파일 기준)의 테스트 기록을
    전부 지운다. 되돌릴 수 없다.

    반환: {"csv_removed": bool, "images_removed": int,
           "students_reset": [학생코드, ...]}
    """
    csv_removed = False
    if SESSIONS_CSV.exists():
        SESSIONS_CSV.unlink()
        csv_removed = True

    images_removed = 0
    if RESULTS_DIR.exists():
        for path in sorted(RESULTS_DIR.rglob("*"), reverse=True):
            if path.is_file():
                if path.name == "보관안내.txt":
                    continue
                path.unlink()
                images_removed += 1
            elif path.is_dir():
                try:
                    path.rmdir()   # 비어 있으면 지워지고, 안 비었으면 그냥 넘어간다
                except OSError:
                    pass

    existing_ids = list(load_students().keys())
    reset_data = {student_id: _blank_state() for student_id in existing_ids}
    save_students(reset_data)

    return {
        "csv_removed": csv_removed,
        "images_removed": images_removed,
        "students_reset": existing_ids,
    }
