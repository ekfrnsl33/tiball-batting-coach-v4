#!/bin/bash
# 티볼 타격 코치 — 카메라 연결 확인
#
# 카메라가 USB 3.0 으로 제대로 붙었는지만 확인합니다.
# 관리자 비밀번호도 필요 없고, 카메라를 건드리지도 않습니다.
# 케이블을 바꿀 때마다 이 파일을 더블클릭해서 확인하세요.

cd "$(dirname "$0")" || exit 1

echo "======================================================================"
echo "  카메라 연결 확인"
echo "======================================================================"
echo ""

TREE=$(ioreg -p IOUSB -w0 -l 2>/dev/null)

if ! echo "$TREE" | grep -q "RealSense"; then
    echo "  ✗ 카메라가 아예 안 보입니다."
    echo ""
    echo "  할 일"
    echo "    1. 케이블이 카메라와 허브(또는 맥북)에 제대로 꽂혔는지 확인"
    echo "    2. 케이블을 뽑고 5초 기다린 뒤 다시 꽂기"
    echo "    3. 허브를 쓰신다면 허브 전원(어댑터)도 확인"
    echo ""
    echo "  이 창은 Enter 를 누르면 닫힙니다."
    read -r _
    exit 1
fi

# RealSense 항목이 나온 뒤 처음 만나는 Device Speed 값을 읽는다.
SPEED=$(echo "$TREE" | awk '/RealSense/{f=1} f&&/"Device Speed"/{gsub(/[^0-9]/,"",$NF); print $NF; exit}')
BCD=$(echo "$TREE" | awk '/RealSense/{f=1} f&&/"bcdUSB"/{gsub(/[^0-9]/,"",$NF); print $NF; exit}')

echo "  카메라를 찾았습니다: Intel RealSense D435i"
echo "    Device Speed = ${SPEED:-?}     bcdUSB = ${BCD:-?}"
echo ""

# macOS의 Device Speed 값
#   2 = High Speed  (480Mbps, USB 2.0)     ← 이러면 촬영 불가
#   3 = SuperSpeed  (5Gbps,  USB 3.0/3.1)  ← 이것부터 정상
#   4 = SuperSpeed+ (10Gbps, USB 3.1 Gen2)
if [ "$SPEED" -ge 3 ] 2>/dev/null; then
    echo "  ✓ USB 3.0 이상으로 연결되었습니다. (SuperSpeed 5Gbps 이상)"
    echo ""
    echo "  이대로 '앱_실행하기.command' 를 더블클릭하시면 됩니다."
else
    echo "  ✗ USB 2.0 으로 연결되었습니다.  이 상태로는 촬영이 안 됩니다."
    echo ""
    echo "  이 카메라는 심도와 색을 동시에 보내야 해서 USB 3.0 이상이 꼭 필요합니다."
    echo "  USB 2.0 으로 붙으면 프로그램이 'No device connected' 만 반복합니다."
    echo ""
    echo "  할 일 - 위에서부터 하나씩 해 보세요"
    echo "    1. 케이블을 허브의 파란색 포트로 옮기기"
    echo "       (또는 SS / 3.0 / 5Gbps 라고 적힌 포트)"
    echo "    2. 카메라 상자에 들어 있던 원래 케이블로 바꾸기"
    echo "       (다른 케이블보다 두껍습니다. 겉모습이 같아도 속 배선이 다릅니다)"
    echo "    3. 랜선 어댑터가 꽂혀 있던 포트에 카메라를 꽂아 보기"
    echo "       (그 포트는 USB 3.2 로 확인되었습니다)"
    echo ""
    echo "  하나 바꿀 때마다 이 파일을 다시 더블클릭해서 확인하세요."
fi

echo ""
echo "======================================================================"
echo "  이 창은 Enter 를 누르면 닫힙니다."
read -r _
