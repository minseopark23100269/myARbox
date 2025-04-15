import numpy as np
import cv2 as cv

# 카메라 캘리브레이션 결과
K = np.array([[754.36932621, 0, 354.80740075],
              [0, 769.75820428, 440.99934945],
              [0, 0, 1]])
dist_coeff = np.array([2.77612703e-01, -2.93292907e+00, 1.32399985e-02, -2.81894361e-03, 9.97246553e+00])

# 체스보드 설정
board_pattern = (8, 6)  # 코너 개수
board_cellsize = 0.025  # 셀 크기

# 구의 좌표 설정 (반지름과 중심)
sphere_radius = board_cellsize * 2  # 구의 반지름
phi, theta = np.meshgrid(np.linspace(0, np.pi, 30), np.linspace(0, 2 * np.pi, 30))
x = sphere_radius * np.sin(phi) * np.cos(theta)
y = sphere_radius * np.sin(phi) * np.sin(theta)
z = sphere_radius * np.cos(phi)
sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T  # 3D 좌표
sphere_center = board_cellsize * np.array([4, 3, 0])  # 체스보드 위 중심 위치
sphere_points += sphere_center

# 비디오 파일 경로
video_file = r'C:\Users\p0105\Videos\chessboard.mp4'
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# 체스보드 탐지 및 구 투영
while True:
    valid, img = video.read()
    if not valid:
        break

    # 체스보드 코너 탐지
    success, img_points = cv.findChessboardCorners(img, board_pattern, None)
    if success:
        # 카메라 포즈 추정
        obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # 구의 좌표를 이미지로 투영
        projected_points, _ = cv.projectPoints(sphere_points, rvec, tvec, K, dist_coeff)
        for point in projected_points:
            cv.circle(img, tuple(np.int32(point.flatten())), 2, (0, 0, 255), -1)  # 빨간색 점

    # 이미지 디스플레이
    cv.imshow('AR Sphere', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()



