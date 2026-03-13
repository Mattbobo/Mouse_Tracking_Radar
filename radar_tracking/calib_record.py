import cv2
import numpy as np
import os

# === 參數設定 ===
pattern_size = (9, 6)        # 棋盤格內部交點數目 (columns, rows)
square_size = 0.028           # 方格邊長 (單位：公尺，例如 0.03 表示 3 cm)
capture_count = 50           # 要收集的有效影像張數

# 準備世界坐標中的棋盤格角點
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 三維點
imgpoints = []  # 二維影像點

# 建立存圖資料夾
os.makedirs('calib_images', exist_ok=True)

cap = cv2.VideoCapture(1)
captured = 0

print("按 'c' 捕捉當前幀，需捕捉 {} 張；按 'q' 退出並進行標定".format(capture_count))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    display = frame.copy()
    if found:
        cv2.drawChessboardCorners(display, pattern_size, corners, found)

    cv2.putText(display, f"Captured: {captured}/{capture_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Calibration', display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and found:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.imwrite(f"calib_images/frame_{captured:02d}.png", frame)
        captured += 1
        print(f"已捕捉: {captured}/{capture_count}")
        if captured >= capture_count:
            break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("捕捉數量不足（至少 5 張），請重新執行。")
else:
    # 相機標定
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    # 儲存為 calib.npz
    np.savez('calib.npz', K=K, dist=dist)
    print("標定完成，檔案已儲存為 'calib.npz'")

