import sys
import cv2
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class VideoThread(QThread):
    change_pixmap = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._pause_flag = False
        self.cap = None
        self.model = None
        self.font = ImageFont.truetype("simhei.ttf", 24)  # 使用黑体
        self.label_map = {"Head": "未佩戴头盔", "Helmet": "头盔", "Person": "人类"}
        self.colors = {"脑袋": (255, 0, 0), "头盔": (0, 255, 0), "人类": (0, 0, 255)}

    def open_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./Models/Hard-Hat-Helmet_01.pt')  # 模型路径

    def run(self):
        self._run_flag = True
        while self._run_flag:
            if not self._pause_flag and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # YOLO检测
                    results = self.model(frame)
                    detections = results.pandas().xyxy[0]
                    
                    # 转换图像为PIL格式处理中文
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    
                    # 绘制检测结果
                    for _, row in detections.iterrows():
                        label = self.label_map.get(row['name'], row['name'])
                        color = self.colors.get(label, (255, 255, 255))
                        
                        # 绘制矩形框
                        draw.rectangle([(row['xmin'], row['ymin']), (row['xmax'], row['ymax'])], 
                                      outline=color, width=2)
                        
                        # 绘制中文标签
                        text = f"{label} {row['confidence']:.2f}"
                        draw.text((row['xmin'], row['ymin']-25), text, font=self.font, fill=color)
                    
                    # 转换回OpenCV格式
                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    
                    self.change_pixmap.emit(frame)
        self.finished.emit()

    def pause(self):
        self._pause_flag = True

    def resume(self):
        self._pause_flag = False

    def stop(self):
        self._run_flag = False
        if self.cap:
            self.cap.release()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        # 创建视频线程
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap.connect(self.update_image)
        # 初始化界面
        self.initUI()

    def initUI(self):
        self.setWindowTitle('YOLO目标检测')
        self.setGeometry(100, 100, 1200, 800)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        self.btn_open = QPushButton('打开视频')
        self.btn_pause = QPushButton('暂停检测')
        self.btn_resume = QPushButton('恢复检测')
        self.btn_close = QPushButton('关闭视频')

        control_layout.addWidget(self.btn_open)
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(self.btn_resume)
        control_layout.addWidget(self.btn_close)
        control_layout.addStretch()

        # 右侧显示区域
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: black;")

        # 添加组件到主布局
        layout.addWidget(control_panel, stretch=1)
        layout.addWidget(self.display_label, stretch=4)

        # 连接信号
        self.btn_open.clicked.connect(self.open_video)
        self.btn_pause.clicked.connect(self.video_thread.pause)
        self.btn_resume.clicked.connect(self.video_thread.resume)
        self.btn_close.clicked.connect(self.close_video)

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开视频文件", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.video_thread.open_video(file_path)
            self.video_thread.start()

    def close_video(self):
        self.video_thread.stop()
        self.display_label.clear()

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.display_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())