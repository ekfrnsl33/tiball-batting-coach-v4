#!/bin/bash
# 티볼 타격 코치 — 강제 종료
#
# 실행 중인 터미널 창을 실수로 닫아버려서 프로그램이 뒤에서 계속 돌아갈 때
# 이 파일을 더블클릭하세요. 관리자 비밀번호를 물어봅니다.
#
# 정상적인 종료 방법은 실행 창에서 Ctrl+C 입니다. 이 파일은 그게 안 될 때만
# 쓰면 됩니다.

echo "======================================================================"
echo "  티볼 타격 코치 — 강제 종료"
echo "======================================================================"
echo ""

if ! pgrep -f "app.py" > /dev/null 2>&1; then
    echo "  실행 중인 프로그램이 없습니다. 종료할 것이 없습니다."
    echo ""
    echo "  이 창은 Enter 를 누르면 닫힙니다."
    read -r _
    exit 0
fi

echo "  실행 중인 프로그램을 찾았습니다. 종료합니다."
echo "  (관리자 비밀번호를 입력해 주세요)"
echo ""

sudo pkill -f "app.py"
sleep 2

if pgrep -f "app.py" > /dev/null 2>&1; then
    echo "  한 번 더 강하게 종료합니다..."
    sudo pkill -9 -f "app.py"
    sleep 1
fi

echo ""
if pgrep -f "app.py" > /dev/null 2>&1; then
    echo "  ⚠ 아직 종료되지 않았습니다. 맥을 다시 시작하면 확실히 정리됩니다."
else
    echo "  ✅ 종료되었습니다. 이제 '앱_실행하기.command' 로 다시 켤 수 있습니다."
fi
echo ""
echo "  이 창은 Enter 를 누르면 닫힙니다."
read -r _
