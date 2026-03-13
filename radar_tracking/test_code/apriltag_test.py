import cv2
import numpy as np

# --------- 參數設定 ---------
# 1) 載入相機標定結果 calib.npz（K 與 dist）
calib = np.load('../calib.npz')
K, dist = calib['K'], calib['dist']

# 2) AprilTag family 與真實邊長 (m)
aruco_dict    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
aruco_params  = cv2.aruco.DetectorParameters()
detector      = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
marker_length = 0.02  # 根據實際尺寸調整

# 3) 打開相機並設定解析度
cap = cv2.VideoCapture(1)  # 如果外接攝影機可能是 0、1
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("按 'q' 鍵退出。")

def normalize(v):
    """向量單位化"""
    return v / np.linalg.norm(v)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # 估算所有標記的 3D 位姿
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, K, dist
        )

        # 收集每個 tag 的位置向量 (X,Y,Z)
        tag_pos = { int(tid): tvec[0] for tid, tvec in zip(ids.flatten(), tvecs) }

        # 只有當 0,1,2,3 四個 tag 都被偵測到時才算
        if all(t in tag_pos for t in (0,1,2,3)):
            O = tag_pos[0]
            # 定義 local X, Y 軸：
            #   X = tag0→tag1（右側）
            #   Y = tag0→tag2（前方，0°）
            x_vec = normalize(tag_pos[1] - O)
            y_vec = normalize(tag_pos[2] - O)
            # 保證正交
            z_vec = normalize(np.cross(x_vec, y_vec))
            y_vec = normalize(np.cross(z_vec, x_vec))
            x_vec = normalize(np.cross(y_vec, z_vec))

            # 計算 tag0→tag3 相對向量
            v = tag_pos[3] - O
            x_proj = np.dot(v, x_vec)
            y_proj = np.dot(v, y_vec)

            # 方位角：atan2(X投影, Y投影)，
            #   當 tag3 在 tag2 前方 (Y>0, X=0) 時 angle=0；
            #   在右側時 angle>0；在左側時 angle<0。
            angle = np.degrees(np.arctan2(x_proj, y_proj))

            # 顯示結果
            cv2.putText(
                frame, f"Azimuth: {angle:.1f}°",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3
            )
            # (可選) 在 console 輸出 debug
            print(f"x_proj={x_proj:.3f}, y_proj={y_proj:.3f}, azimuth={angle:.1f}°")

        # 在影像上繪製坐標軸
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, marker_length*0.5)

    cv2.imshow("AprilTag Azimuth", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
