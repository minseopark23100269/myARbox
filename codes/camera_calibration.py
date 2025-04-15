import numpy as np
import cv2 as cv

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    img_select = []
    while True:
        # Grab an image from the video
        valid, img = video.read()
        if not valid:
            break

        # Debugging: Check chessboard detection
        complete, pts = cv.findChessboardCorners(img, board_pattern)
        if not complete:
            print("체스보드를 감지하지 못했습니다.")
        else:
            print("체스보드 감지 성공!")
            cv.drawChessboardCorners(img, board_pattern, pts, complete)
        
        # Add image if chessboard is detected
        if complete:
            img_select.append(img)
            print(f"이미지가 선택되었습니다! 현재 선택된 이미지 수: {len(img_select)}")

        # If not select_all, allow interactive selection
        if not select_all:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == ord(' '):  # Space: Pause and show corners
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):  # Enter: Select the image
                    img_select.append(img)
            if key == 27:  # ESC: Exit (Complete image selection)
                break

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0, 'There is no set of complete chessboard points!'

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)  # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

if __name__ == '__main__':
    video_file = r'C:\Users\p0105\Videos\chessboard.mp4'  # .mp4 파일 경로
    board_pattern = (8, 6)  # 내부 코너 개수 (열, 행)
    board_cellsize = 0.025  # 셀 크기 (미터 단위)

    img_select = select_img_from_video(video_file, board_pattern, select_all=True)
    assert len(img_select) > 0, 'There is no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')


