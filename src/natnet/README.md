# NatNet ROS2 Integration

OptiTrack Motion Capture System용 ROS2 패키지입니다. Motive에서 실시간으로 모션 캡처 데이터를 받아 ROS2 토픽으로 발행합니다.

## 구성 요소

### NatNetClient
OptiTrack Motive 소프트웨어와 NatNet 프로토콜로 통신하는 Python 클라이언트입니다.

**주요 기능:**
- Motive 서버로부터 실시간 모션 캡처 프레임 수신
- Rigid Body 데이터 파싱 (위치, 회전, 추적 상태)
- Unlabeled Marker 데이터 파싱
- NatNet 프로토콜 버전 자동 협상 (2.x ~ 4.x 지원)

**데이터 수신 방식:**
- UDP 소켓을 통한 실시간 스트리밍
- Unicast 또는 Multicast 모드 지원
- Command 채널 (포트 1510): 서버 정보, 데이터 모델 요청
- Data 채널 (포트 1511): 모션 캡처 프레임 데이터

### OptiRos
NatNetClient를 래핑하여 ROS2 노드로 동작하는 인터페이스입니다.

**주요 기능:**
- NatNet 데이터를 ROS2 메시지로 변환
- 실시간 데이터 발행 (30Hz 기본)
- Rigid Body 및 Unlabeled Marker 시각화 지원

## ROS2 토픽

### 1. `/natnet_client_node/marker_array` (visualization_msgs/MarkerArray)

**데이터 소스:** Motive에서 정의된 Rigid Body (강체 객체)

**발행 주기:** 실시간 (모션 캡처 프레임 수신 시마다)

**데이터 형식:**
```python
MarkerArray:
  markers: [Marker]
    - header:
        frame_id: "opti_world"
        stamp: ROS time
    - ns: str(rigid_body_id)
    - id: int(rigid_body_id)
    - type: CUBE (5)
    - action: ADD (0)
    - pose:
        position: Point(x, y, z)      # 미터 단위
        orientation: Quaternion(x, y, z, w)  # 회전
    - scale: Vector3(0.05, 0.05, 0.05)  # 5cm 큐브
    - color: ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # 초록색
    - lifetime: 0.1초
```

**데이터 특성:**
- 각 Rigid Body는 고유 ID를 가지며, Motive에서 정의한 마커 세트입니다
- 추적이 유효한(tracking_valid=True) Rigid Body만 발행됩니다
- 위치는 Motive의 좌표계 기준 (일반적으로 Y-up)
- 회전은 쿼터니언으로 표현됩니다

**사용 예시:**
- RViz2에서 `/natnet_client_node/marker_array`를 구독하여 Rigid Body 시각화
- 로봇의 위치 추적
- 다중 객체 추적 및 제어

---

### 2. `/natnet_client_node/unlabeled` (visualization_msgs/MarkerArray)

**데이터 소스:** Motive에서 Rigid Body에 할당되지 않은 개별 마커들

**발행 주기:** 30Hz (타이머 기반)

**데이터 형식:**
```python
MarkerArray:
  markers: [Marker]
    - header:
        frame_id: "opti_world"
        stamp: ROS time
    - id: int(marker_index)  # 0부터 시작하는 순차 인덱스
    - type: SPHERE (2)
    - action: ADD (0)
    - pose:
        position: Point(x, y, z)  # 마커의 3D 위치 (미터)
        orientation: Quaternion(0, 0, 0, 1)  # 기본 방향
    - scale: Vector3(0.02, 0.02, 0.02)  # 2cm 구체
    - color: ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # 빨간색
    - lifetime: 0.1초
```

**⚠️ 중요: Residual 의존성**

Unlabeled Marker 데이터는 **Motive의 Residual 설정에 크게 영향을 받습니다:**

1. **Residual이란?**
   - 마커의 3D 재구성 오차 (단위: mm)
   - 여러 카메라에서 본 2D 위치를 3D로 변환할 때의 불확실성
   - 낮은 값일수록 정확한 추적을 의미

2. **발행 조건:**
   - Motive에서 설정한 **Residual Threshold 이하**의 마커만 발행됩니다
   - 예: Residual < 1.0mm로 설정 시, 1.0mm 이상의 오차를 가진 마커는 제외
   
3. **Residual 설정 확인 방법:**
   ```
   Motive → View → Data Streaming Pane
   → Advanced Network Settings
   → "Send Unlabeled Markers" 체크
   → Residual Threshold 확인 (기본값: 1.0mm)
   ```

4. **실전 팁:**
   - 마커가 보이지 않는 경우 → Residual Threshold를 높여보세요 (2.0~5.0mm)
   - 너무 높은 Threshold → 부정확한 마커 위치가 포함될 수 있음
   - 최적값은 환경에 따라 다름 (조명, 카메라 배치, 캘리브레이션 품질)

5. **디버깅:**
   - Motive의 3D View에서 개별 마커의 Residual 값 확인 가능
   - 마커를 선택하면 Properties 패널에서 실시간 Residual 표시
   - 일반적으로 0.3~0.8mm는 매우 좋은 추적 품질

**사용 예시:**
- 개별 마커 위치 추적
- 실시간 마커 포인트 클라우드 생성
- 커스텀 강체 정의 (Rigid Body를 Motive에서 정의하지 않고 ROS에서 처리)
- 마커 기반 SLAM 또는 로컬라이제이션

---

## 설정 및 실행

### 네트워크 설정

```python
# OptiRos.py의 optionsDict 수정
optionsDict = {
    "clientAddress": "192.168.50.30",  # 이 컴퓨터의 IP
    "serverAddress": "192.168.50.45",  # Motive 실행 중인 PC의 IP
    "use_multicast": False,             # Unicast 사용
    "stream_type": "d",                 # Data stream
}
```

### Motive 설정 확인

1. **Data Streaming 활성화:**
   - `View → Data Streaming Pane`
   - "Broadcast Frame Data" 체크
   - "Stream Rigid Bodies" 체크
   - "Send Unlabeled Markers" 체크 (Unlabeled Marker 발행 시 필수)

2. **네트워크 설정:**
   - Type: Unicast (권장) 또는 Multicast
   - Local Interface: Motive PC의 네트워크 인터페이스
   - Command Port: 1510
   - Data Port: 1511

3. **좌표계 설정:**
   - `Edit → Preferences → Streaming`
   - Up Axis: Y (OpenGL 기본)

### 실행

```bash
# 빌드
cd /home/jack/project_SORO
colcon build

# 소스
source install/setup.bash

# 실행
python3 src/natnet/natnet/OptiRos.py
```

### RViz2 시각화

```bash
# 다른 터미널에서
rviz2
```

**RViz2 설정:**
1. Fixed Frame: `opti_world`
2. Add → MarkerArray
   - Topic: `/natnet_client_node/marker_array` (Rigid Bodies)
3. Add → MarkerArray
   - Topic: `/natnet_client_node/unlabeled` (Unlabeled Markers)

---

## 아키텍처

```
┌─────────────────┐
│ Motive Software │
│  (OptiTrack)    │
└────────┬────────┘
         │ NatNet Protocol (UDP)
         │ Port 1510 (Command)
         │ Port 1511 (Data)
         ▼
┌─────────────────────────┐
│    NatNetClient.py      │
│  - 프로토콜 파싱        │
│  - 데이터 언팩          │
│  - 버전 협상            │
└────────┬────────────────┘
         │ Python Callbacks
         │ - rigid_body_listener
         │ - new_frame_listener
         ▼
┌─────────────────────────┐
│      OptiRos.py         │
│  (ROS2 Node)            │
│  - 메시지 변환          │
│  - 타이머 기반 발행     │
└────────┬────────────────┘
         │ ROS2 Topics
         ├─→ /natnet_client_node/marker_array
         └─→ /natnet_client_node/unlabeled
         ▼
┌─────────────────────────┐
│  ROS2 Subscribers       │
│  - RViz2                │
│  - 사용자 노드          │
└─────────────────────────┘
```

---

## 데이터 플로우

### Rigid Body 데이터

```
Motive Rigid Body 
  ↓
NatNetClient.__unpack_rigid_body()
  ↓ (id, position, rotation)
rigid_body_listener callback
  ↓
OptiRos._get_marker_msg()
  ↓ (Marker 메시지 생성)
OptiRos._update_marker_array()
  ↓ (MarkerArray에 추가/업데이트)
OptiRos._publish_marker_array()
  ↓
/natnet_client_node/marker_array 토픽
```

### Unlabeled Marker 데이터

```
Motive Unlabeled Markers (Residual < Threshold)
  ↓
NatNetClient.__unpack_marker_set_data()
  ↓ (position list)
NatNetClient._unlabeled_marker_array 저장
  ↓
OptiRos._publish_marker_array() (30Hz 타이머)
  ↓
/natnet_client_node/unlabeled 토픽
```

---

## 트러블슈팅

### 1. 데이터가 수신되지 않음
- Motive에서 "Broadcast Frame Data" 활성화 확인
- 방화벽에서 UDP 포트 1510, 1511 허용
- IP 주소가 올바른지 확인 (`ifconfig` 또는 `ip addr`)
- 같은 네트워크 세그먼트에 있는지 확인

### 2. Unlabeled Marker가 보이지 않음
- **Residual Threshold 확인 및 조정** (가장 흔한 원인)
- Motive에서 "Send Unlabeled Markers" 체크 확인
- Motive 3D View에서 마커가 실제로 추적되는지 확인
- 마커 개수가 0이 아닌지 확인: `Unlabeled Marker Count: X` 로그 확인

### 3. Rigid Body가 추적되지 않음
- Motive에서 Rigid Body가 정의되어 있는지 확인
- 최소 3개 이상의 마커로 Rigid Body 구성 필요
- "Stream Rigid Bodies" 활성화 확인
- Tracking 상태가 Valid인지 확인 (녹색 표시)

### 4. 좌표계가 이상함
- Motive의 Up Axis 설정 확인 (Y-up 권장)
- RViz2의 Fixed Frame이 `opti_world`인지 확인
- 필요시 OptiRos.py에서 좌표 변환 추가

### 5. 성능 이슈
- 발행 주기 조정: `self._hz` 값 변경 (기본 30Hz)
- 불필요한 마커 필터링
- RViz2의 MarkerArray History Length 조정

---

## 주요 파라미터

### NatNetClient
- `server_ip_address`: Motive 서버 IP
- `local_ip_address`: 클라이언트 IP
- `command_port`: 1510 (기본값)
- `data_port`: 1511 (기본값)
- `use_multicast`: Multicast/Unicast 선택

### OptiRos
- `_hz`: 발행 주기 (기본 30.0Hz)
- Frame ID: `"opti_world"` (모든 메시지의 좌표계)

---

## 좌표계 정보

**OptiTrack 좌표계 (기본):**
- X: 오른쪽
- Y: 위쪽 (Up)
- Z: 앞쪽
- 단위: 미터 (m)
- 회전: 쿼터니언 (x, y, z, w)

**ROS REP-103 표준 변환이 필요한 경우:**
- ROS 표준: X(전방), Y(좌측), Z(상방)
- 변환 코드를 OptiRos.py에 추가 필요

---

## 라이선스

Copyright © 2025 NaturalPoint, Inc. All Rights Reserved.

OptiTrack Plugins EULA: https://www.optitrack.com/about/legal/eula.html

---

## 참고 자료

- [OptiTrack Documentation](https://docs.optitrack.com/)
- [NatNet SDK](https://docs.optitrack.com/developer-tools/natnet-sdk)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [visualization_msgs](https://docs.ros.org/en/humble/p/visualization_msgs/)
