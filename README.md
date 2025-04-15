# myARbox
A program for camera pose estimation and AR object visualization using OpenCV


# 카메라 자세 추정 및 AR 구현

## 프로그램 설
이 프로그램은 체스보드를 사용하여 카메라의 자세를 추정하고, 영상 위에 증강 현실(AR) 물체를 시각화하는 프로젝트입니다. OpenCV를 활용하여 간단한 도형, 숫자/알파벳, 또는 3D 모델을 체스보드 위에 표시할 수 있습니다. 저의 경우는 구(공 모양)를 시각화했습니다. 

## 목표
- **카메라 자세 추정**: 체스보드 영상과 카메라 캘리브레이션 데이터를 기반으로 카메라의 위치와 자세를 계산합니다.
- **AR 물체 표시**: 계산된 카메라 자세를 활용해 체스보드 위에 AR 물체를 렌더링합니다.

## 기능
1. **카메라 캘리브레이션**:
   - 체스보드 패턴 기반으로 카메라 내부 파라미터와 왜곡 계수를 계산.
   - 이전 과제(Homework #3)의 결과를 활용했습니다. 

2. **AR 물체 시각화**:
   - 구를 영상에 렌더링.
   - OpenCV를 활용한 효율적인 이미지 처리.

3. **동영상 처리**:
   - 사전 녹화된 체스보드 영상으로 AR 렌더링.





