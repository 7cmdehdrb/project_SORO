# Arduino Pneumatic Control System

이 폴더는 아두이노 기반 공압 제어 시스템을 위한 파일들을 포함합니다.

## 파일 구성

### 1. `pneumatic_pid_control.ino`
아두이노 펌웨어로, 시리얼 통신을 통해 공압 밸브를 제어합니다.

**핀 구성:**
- `solenoidPin` (7번): 솔레노이드 밸브 제어 (디지털 출력)
- `proportionalPin` (8번): 비례 밸브 제어 (PWM 출력)
- `sensorPin` (A0번): 압력 센서 입력 (아날로그 입력)

**통신 프로토콜:**
- 시리얼 통신: 9600 baud
- 패킷 크기: 10바이트 고정

**패킷 구조:**
```
[SOF][CMD][I32(4)][FLAGS(1)][CRC8(1)][EOF]
```

| 바이트 | 필드 | 설명 |
|--------|------|------|
| 0 | SOF | Start of Frame (0xAA) |
| 1 | CMD | 명령 코드 (예: 0x01) |
| 2-5 | I32 | 32비트 정수 (little-endian), 실제값 = I32 / 1000 |
| 6 | FLAGS | bit0 = 불린 값 (0/1), 나머지 비트는 예약 |
| 7 | CRC8 | CRC8 체크섬 (CMD + I32 + FLAGS) |
| 8 | EOF | End of Frame (0x55) |
| 9 | (종료) | 총 10바이트 |

**제어 동작:**
- FLAGS의 bit0 → 솔레노이드 밸브 ON/OFF
- I32 스케일링 값 → 비례 밸브 PWM (0-255)

### 2. `send_pneumatic_control.py`
아두이노로 제어 패킷을 전송하는 파이썬 스크립트입니다.

**주요 클래스:**
- `PneumaticControlSender`: 시리얼 통신 및 패킷 생성 클래스

**주요 메서드:**
- `create_packet(cmd, float_value, bool_value)`: 제어 패킷 생성
- `send_packet(cmd, float_value, bool_value)`: 패킷 전송
- `send_random(cmd)`: 랜덤 값으로 패킷 전송
- `calc_crc8(data)`: CRC8 체크섬 계산

**기능:**
- 랜덤 불린 값 (True/False) 생성 → 솔레노이드 밸브 제어
- 랜덤 실수 값 (0.0~255.0) 생성 → 비례 밸브 제어
- 1초마다 제어 패킷 자동 전송

## 사용 방법

### 1. 아두이노 설정
```bash
# 아두이노 IDE에서 pneumatic_pid_control.ino를 열어 업로드
```

### 2. 파이썬 환경 설정
```bash
# pyserial 라이브러리 설치
pip install pyserial
```

### 3. 파이썬 스크립트 실행
```bash
python send_pneumatic_control.py
```

**COM 포트 변경:**
```python
# send_pneumatic_control.py의 main() 함수에서
sender = PneumaticControlSender(port='COM3', baudrate=9600)
# COM3를 실제 아두이노 포트로 변경
```

**포트 확인 방법:**
- Windows: 장치 관리자 → 포트(COM & LPT) 확인
- Linux: `ls /dev/ttyUSB*` 또는 `ls /dev/ttyACM*`
- macOS: `ls /dev/cu.*`

## 프로토콜 세부사항

### CRC8 알고리즘
- 폴리노미얼: 0x07
- 초기값: 0x00
- 계산 범위: CMD + I32 + FLAGS (5바이트)

### 스케일링
- SCALE = 1000
- 실제 값 = I32 / 1000
- 예: I32 = 123456 → 실제 값 = 123.456

### 플래그 비트
- bit0: 불린 제어 값 (0 = OFF, 1 = ON)
- bit1-7: 예약 (항상 0)

## 예제

### 수동 패킷 전송
```python
from send_pneumatic_control import PneumaticControlSender

sender = PneumaticControlSender(port='COM3', baudrate=9600)

# 솔레노이드 ON, 비례 밸브 150 설정
sender.send_packet(cmd=0x01, float_value=150.5, bool_value=True)

# 솔레노이드 OFF, 비례 밸브 0 설정
sender.send_packet(cmd=0x01, float_value=0.0, bool_value=False)

sender.close()
```

### 랜덤 테스트
```python
from send_pneumatic_control import PneumaticControlSender
import time

sender = PneumaticControlSender(port='COM3', baudrate=9600)

# 10번 랜덤 패킷 전송
for i in range(10):
    sender.send_random(cmd=0x01)
    time.sleep(0.5)

sender.close()
```

## 문제 해결

### 아두이노가 응답하지 않음
1. COM 포트가 올바른지 확인
2. 아두이노가 연결되어 있는지 확인
3. 다른 프로그램이 시리얼 포트를 사용 중인지 확인
4. 아두이노 IDE의 시리얼 모니터를 닫았는지 확인

### 패킷이 전송되지 않음
1. pyserial 라이브러리 설치 확인: `pip list | grep pyserial`
2. 방화벽/보안 프로그램 확인
3. USB 케이블 연결 상태 확인

### 밸브가 동작하지 않음
1. 아두이노 핀 연결 확인
2. 전원 공급 확인
3. 패킷의 CRC8이 올바른지 확인 (디버깅 출력 확인)

## 라이선스
이 프로젝트의 라이선스는 상위 디렉토리의 LICENSE 파일을 참조하세요.
