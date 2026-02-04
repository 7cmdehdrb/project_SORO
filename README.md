# STEP 1: State Estimation Model 구축

본 단계는 전체 프로젝트의 일부로서, 소프트 로봇의 상태(State)를 추정하기 위한 학습 기반 모델 구축을 목표로 한다. 로봇에 무작위 액추에이션 입력을 가하고, 이에 따른 센서 출력 데이터를 수집·학습하여 입력 대비 상태를 예측하는 모델을 생성한다.

---

## 1. 목표

- 소프트 로봇에 **허용 가능한 범위 내의 랜덤 액추에이션 입력**을 인가한다.
- 입력 데이터와 출력 데이터를 체계적으로 **로깅**한다.
- 수집된 데이터를 이용하여 **상태 추정 모델(State Estimation Model)**을 학습한다.
- 초기 모델 구조로 **LSTM 기반 시계열 모델**을 사용한다.

---

## 2. 전체 파이프라인 개요

1. 랜덤 액추에이션 입력 생성  
2. 로봇 구동 및 센서 출력 획득  
3. 입력/출력 데이터 로깅  
4. 학습 모델 구축  
5. 성능 평가

---

## 3. 액추에이션 입력 생성

- 입력은 소프트 로봇의 물리적 한계를 초과하지 않는 범위에서 생성한다.
- 랜덤 입력은 다음 조건을 만족해야 한다.
  - 시스템 파손 위험 없음
  - 센서 응답이 관측 가능한 수준
  - 시간적으로 연속성을 가지는 시계열 형태

예시:
- 압력 입력  
- 모터 구동 신호  
- 공기 주입량 등  

---

## 4. 데이터 로깅

### 4.1 입력 데이터
- 액추에이터 제어 신호
- 타임스탬프

### 4.2 출력 데이터
- 센서 측정값 (예: 변형, 위치, 압력, 각도 등)
- 타임스탬프

### 4.3 저장 형식
- CSV 또는 SQL 데이터베이스 형식 사용
- 입력과 출력은 동일한 시간 기준으로 정렬하여 저장

---

## 5. 학습 모델

### 5.1 모델 개요
- 모델 구조: LSTM 기반 시계열 회귀 모델
- 목적: 과거 입력 및 센서 데이터를 기반으로 현재 상태 추정

### 5.2 입력
- 액추에이션 시퀀스
- 센서 시퀀스 (필요 시)

### 5.3 출력
- 추정된 로봇 상태 벡터

---

# STEP 1: State Estimation Model

This step is a part of the overall project and aims to build a learning-based model for estimating the state of a soft robot. Random actuation inputs are applied to the robot, and corresponding sensor outputs are collected and used to train a model that predicts the robot state from the inputs.

---

## 1. Objective

- Apply **random actuation inputs within acceptable limits** to the soft robot.
- Systematically **log input and output data**.
- Train a **state estimation model** using the collected dataset.
- Use an **LSTM-based time-series model** as the initial model architecture.

---

## 2. Overall Pipeline

1. Generate random actuation inputs  
2. Drive the robot and acquire sensor outputs  
3. Log input/output data  
4. Build a learning model  
5. Evaluate performance  

---

## 3. Actuation Input Generation

- Inputs must remain within the physical limits of the soft robot.
- Random inputs should satisfy the following conditions:
  - No risk of system damage  
  - Sensor responses must be observable  
  - Time-continuous sequence structure  

Examples:
- Pressure input  
- Motor control signals  
- Air injection volume  

---

## 4. Data Logging

### 4.1 Input Data
- Actuator control signals  
- Timestamps  

### 4.2 Output Data
- Sensor measurements (e.g., deformation, position, pressure, angle)  
- Timestamps  

### 4.3 Storage Format
- CSV or SQL database format  
- Inputs and outputs must be synchronized based on timestamps  

---

## 5. Learning Model

### 5.1 Model Overview
- Architecture: LSTM-based time-series regression model  
- Purpose: Estimate the current state from past inputs and sensor data  

### 5.2 Inputs
- Actuation sequences  
- Sensor sequences (optional)  

### 5.3 Outputs
- Estimated robot state vector  

---
