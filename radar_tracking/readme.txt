# Radar & Camera Recorder

整合毫米波雷達與攝影機影像的同步錄製工具，支援 AprilTag 自動校準、角度追蹤、特徵圖儲存、影片輸出與 CSV 標註生成。


#calib_record

準備好標定板，生成calib.npz來校準相機誤差

## 📦 功能說明

- 📡 毫米波雷達資料接收（KKT Module）
- 📷 攝影機即時畫面串流（OpenCV）
- 🏷️ AprilTag 3D Pose 偵測 + ID 0,1,2 自動校準角度基準
- 📐 實時顯示 ID=3 的相對水平角度
- 💾 錄製功能：
  - 同步存下雷達特徵圖（`record_xxx.h5`）
  - 攝影機錄影（`record_xxx.mp4`）
  - 每幀角度輸出（`angles.csv`）

## 🧰 環境需求

- Python 3.8+
- OpenCV (`cv2`)
- numpy
- h5py
- PySide2
- 已安裝並配置 `KKT_Module` 與 `KKT_UI` 相關模組

🧱 場景佈置建議
為了獲得準確的角度校準與測量，請依照以下建議進行場景佈置：

📐 基準點（AprilTag ID 0、1、2）
使用尺寸相同的 AprilTag（DICT_APRILTAG_16H5），建議邊長為 4 公分。

水平放置在相機可見範圍內，左右對稱分布：
Tag0：中央底部（作為原點）
Tag1：tag0右側（對應 90°）
Tag2：tag0前方（對應 0°）
確保這三個標籤在相機畫面中同時可見，且不要傾斜。


🎯 目標點（AprilTag ID 3）
目標物應貼上 Tag3，放在測量區中任一角度位置。
鏡頭將以此 ID 作為追蹤目標，顯示其相對於基準的水平角度。

📷 相機位置與角度
相機建議擺放於基準點正前方，高度略高於 AprilTag 水平中心，並盡量保持水平不傾斜。
避免傾斜或歪斜的角度，這會使得 3D 偵測失準。

