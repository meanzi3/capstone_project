import cv2
import sys
import numpy as np
import mediapipe as mp
import keras
from PyQt5.QtCore import pyqtProperty

import state
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QSizePolicy, QDesktopWidget
from landmark import mp_drawing, mp_holistic, Frame
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore

running = False
start_capture = True

# Variable to use if current_state in (state.ORDER_MENU, state.ORDER_OPTION)
actions = np.array(['americano', 'cafelatte', 'cafemocha', 'ice', 'hot'])
actions_ko = {
    'menu': {
        'americano': '아메리카노',
        'cafelatte': '카페라떼',
        'cafemocha': '카페모카'
    },
    'option': {
        'ice': '아이스',
        'hot': '핫'
    }
}
sentence = []

# Variable to use if current_state == state.ORDER_COUNT
gesture = {
    0: -1, 1: 1, 2: -1, 3: 3, 4: 4, 5: 5,
    6: -1, 7: -1, 8: -1, 9: 2, 10: -1, 11: -1
}
gesture_count_threshold = 60
gesture_counts = {i: 0 for i in range(len(gesture))}
gesture_displayed = False

current_state = state.ORDER_NONE
order_list = []

class CamThread(QtCore.QThread):
    orderData = QtCore.pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

    def run(self):
        global running
        global actions, start_capture, sentence
        global gesture, gesture_count_threshold, gesture_counts, gesture_displayed
        global current_state

        # Variable to use if current_state in (state.ORDER_MENU, state.ORDER_OPTION)
        sequence = []
        threshold = 0.9  # 적중 확률 임계값.
        circle_color = (0, 255, 0)  # 녹화 표시 원. 붉은색=녹화중.
        model = keras.models.load_model('data/action.h5')
        res = np.zeros(len(actions))

        # Variable to use if current_state == state.ORDER_COUNT
        max_num_hands = 1
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        file = np.genfromtxt('data/gesture_train_fy.csv', delimiter=',')
        angle = file[:, :-1].astype(np.float32)
        label = file[:, -1].astype(np.float32)
        knn = cv2.ml.KNearest_create()
        knn.train(angle, cv2.ml.ROW_SAMPLE, label)

        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print('width: {}, height: {}'.format(width, height))
        # self.parent.label_cam.resize(width, height)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while running:
                if start_capture:
                    ret, img = cap.read()
                    if ret:
                        if current_state in (state.ORDER_MENU, state.ORDER_OPTION):
                            frame = Frame(img)
                            # Make detections
                            results = frame.mediapipe_detection(holistic)
                            print(results)
                            frame.set_image(cv2.flip(frame.get_image(), 1))
                            # 랜드마크 생성
                            frame.draw_styled_landmarks(results)
                            if start_capture:
                                keypoints = frame.extract_keypoints(results)
                                sequence.append(keypoints)
                                print(len(sequence))
                                circle_color = (0, 0, 255)  # 추출 중이므로 붉은 원

                                # 프레임 길이가 120이 되면 메뉴 예측 및 주문 진행
                                if len(sequence) == 120:
                                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                                    # print(actions[np.argmax(res)])
                                    if res[np.argmax(res)] > threshold:
                                        if len(sentence) > 0 and actions[np.argmax(res)] != sentence[-1]:
                                            sentence.append(actions[np.argmax(res)])
                                        elif len(sentence) == 0:
                                            sentence.append(actions[np.argmax(res)])
                                    else:
                                        sentence.append("failed")

                                    if len(sentence) > 4:
                                        sentence = sentence[-4:]

                                    start_capture = False
                                    circle_color = (0, 255, 0)
                                    sequence = []

                                    # check order
                                    self.orderData.emit(str(sentence[-1]))
                                    if current_state == state.ORDER_MENU:
                                        self.parent.set_order_visible(False)
                                        if sentence[-1] in actions_ko['menu']:
                                            self.parent.set_check_visible(True)
                                        else:
                                            self.parent.set_failed_visible(True)
                                    elif current_state == state.ORDER_OPTION:
                                        self.parent.set_option_visible(False)
                                        if sentence[-1] in actions_ko['option']:
                                            self.parent.set_check_visible(True)
                                        else:
                                            self.parent.set_failed_visible(True)

                            # drawed_img = frame.prob_viz(res, actions, frame.get_image())
                            drawed_img = frame.get_image().copy()
                            # cv2.circle(drawed_img, (1200, 50), 20, circle_color, -1)
                            cv2.putText(drawed_img, str(len(sequence)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, circle_color, 2, cv2.LINE_AA)

                            # cv2.rectangle(drawed_img, (0, 0), (640, 40), (245, 117, 16), -1)
                            # cv2.putText(drawed_img, ' '.join(sentence), (3, 30),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                            drawed_img = cv2.cvtColor(drawed_img, cv2.COLOR_BGR2RGB)
                            h, w, c = drawed_img.shape
                            qImg = QtGui.QImage(drawed_img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                            pixmap = QtGui.QPixmap.fromImage(qImg)
                            pixmap = pixmap.scaledToHeight(int(self.parent.screen_size.height() * 0.45))
                            self.parent.label_cam.setPixmap(pixmap)

                        elif current_state == state.ORDER_COUNT:
                            # todo: hand gesture detection
                            drawed_img = cv2.flip(img, 1)
                            drawed_img = cv2.cvtColor(drawed_img, cv2.COLOR_BGR2RGB)

                            result = hands.process(drawed_img)

                            drawed_img = cv2.cvtColor(drawed_img, cv2.COLOR_RGB2BGR)

                            if result.multi_hand_landmarks is not None:
                                for res in result.multi_hand_landmarks:
                                    joint = np.zeros((21, 3))
                                    for j, lm in enumerate(res.landmark):
                                        joint[j] = [lm.x, lm.y, lm.z]

                                    # Compute angles between joints
                                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                                         :]  # Parent joint
                                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                         :]  # Child joint
                                    v = v2 - v1  # [20,3]
                                    # Normalize v
                                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                                    # Get angle using arcos of dot product
                                    angle = np.arccos(np.einsum('nt,nt->n',
                                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18],
                                                                :],
                                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                                :]))  # [15,]

                                    angle = np.degrees(angle)  # Convert radian to degree

                                    # Inference gesture
                                    data = np.array([angle], dtype=np.float32)
                                    ret, results, neighbours, dist = knn.findNearest(data, 3)
                                    idx = int(results[0][0])

                                    mp_drawing.draw_landmarks(drawed_img, res, mp_hands.HAND_CONNECTIONS)

                                    # Update gesture counts
                                    gesture_counts[idx] += 1

                                    # Check if any gesture count has reached the threshold
                                    for i, count in gesture_counts.items():
                                        if count >= gesture_count_threshold and not gesture_displayed:
                                            print(f"Gesture {gesture[i]} occurred {count} times.")
                                            final_gesture = gesture[i]

                                            self.orderData.emit(str(final_gesture))
                                            self.parent.set_count_visible(False)
                                            if final_gesture in range(1, 6):
                                                self.parent.set_check_visible(True)
                                            else:
                                                self.parent.set_failed_visible(True)

                                            gesture_displayed = True
                                            break  # Exit the loop after displaying the gesture
                            drawed_img = cv2.cvtColor(drawed_img, cv2.COLOR_BGR2RGB)
                            h, w, c = drawed_img.shape
                            qImg = QtGui.QImage(drawed_img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                            pixmap = QtGui.QPixmap.fromImage(qImg)
                            pixmap = pixmap.scaledToHeight(int(self.parent.screen_size.height() * 0.45))
                            self.parent.label_cam.setPixmap(pixmap)

                    else:
                        QtWidgets.QMessageBox.about(self.parent.win, 'Error', 'Cannot read frame.')
                        print('cannot read frame.')
                        break
            cap.release()
            print('Thread end.')

    def checkOrder(action):
        print(action)


class MainWidget(QtWidgets.QWidget):
    def __init__(self, theme='dark'):
        global app
        super().__init__()

        self.screen_size = app.desktop().screenGeometry()

        self.theme = theme
        QtGui.QFontDatabase().addApplicationFont("data/font/잘풀리는오늘 Medium.ttf")

        self.last_order = dict()
        self.thread_cam = CamThread(self)
        self.thread_cam.orderData.connect(self.set_ordered_menu)

        with open('style.qss', 'r') as f:
            self.setStyleSheet(f.read())
        # apply_stylesheet(self, theme='dark_lightgreen.xml')

        # 시작 화면
        self.font_big = QFont('잘풀리는오늘 Medium')
        self.font_big.setPointSize(32)
        # self.font_big.setBold(True)
        self.font_medium = QFont('잘풀리는오늘 Medium')
        self.font_medium.setPointSize(24)
        self.font_small = QFont('잘풀리는오늘 Medium')
        self.font_small.setPointSize(16)

        self.label_welcome = QtWidgets.QLabel()
        self.label_welcome.setAlignment(QtCore.Qt.AlignCenter)
        self.label_welcome.setFont(self.font_big)
        self.pixmap_logo = QtGui.QPixmap()
        self.pixmap_logo.load('data/image/logo3.png')
        self.pixmap_logo = self.pixmap_logo.scaledToHeight(int(self.screen_size.height() * 0.6))
        self.label_welcome.setPixmap(self.pixmap_logo)

        self.btn_start = QtWidgets.QPushButton(' 주문 시작')
        self.btn_start.setFont(self.font_big)
        self.icon_start = QtGui.QIcon('data/image/btn_start_icon.png')
        self.btn_start.setIcon(self.icon_start)
        self.btn_start.setIconSize(QtCore.QSize(120, 120))
        self.btn_start.setObjectName('btn_start')
        self.btn_start.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_start.clicked.connect(self.start)

        # 주문 화면
        self.label_menu_ment = QtWidgets.QLabel('화면에 나온 수어를 참고해 음료를 주문해 주세요.')
        self.label_menu_ment.setAlignment(QtCore.Qt.AlignCenter)
        self.label_menu_ment.setFont(self.font_medium)
        self.label_menu_ment.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.label_menu_ment.setObjectName('label_ment')

        self.label_cam = QtWidgets.QLabel()
        self.label_cam.setAlignment(QtCore.Qt.AlignCenter)
        self.label_cam.setObjectName('label_cam')

        self.vbox_menu = {
            'americano': QtWidgets.QVBoxLayout(),
            'cafelatte': QtWidgets.QVBoxLayout(),
            'cafemocha': QtWidgets.QVBoxLayout()
        }
        self.frame_menu = {
            'americano': QtWidgets.QFrame(),
            'cafelatte': QtWidgets.QFrame(),
            'cafemocha': QtWidgets.QFrame()
        }
        self.hbox_menu = QtWidgets.QHBoxLayout()
        self.hbox_menu.setAlignment(QtCore.Qt.AlignCenter)

        self.label_menu = {
            'sign_americano': QtWidgets.QLabel(),
            'sign_cafelatte': QtWidgets.QLabel(),
            'sign_cafemocha': QtWidgets.QLabel(),
            'text_americano': QtWidgets.QLabel('아메리카노'),
            'text_cafelatte': QtWidgets.QLabel('카페라떼'),
            'text_cafemocha': QtWidgets.QLabel('카페모카')
        }
        self.movie_menu = {
            'americano': QtGui.QMovie('data/아메리카노.gif'),
            'cafelatte': QtGui.QMovie('data/카페라떼.gif'),
            'cafemocha': QtGui.QMovie('data/카페모카.gif')
        }
        self.movie_menu['americano'].setScaledSize(
            QtCore.QSize(int(self.screen_size.height() * 0.2), int(self.screen_size.height() * 0.2)))
        self.movie_menu['cafelatte'].setScaledSize(
            QtCore.QSize(int(self.screen_size.height() * 0.2), int(self.screen_size.height() * 0.2)))
        self.movie_menu['cafemocha'].setScaledSize(
            QtCore.QSize(int(self.screen_size.height() * 0.2), int(self.screen_size.height() * 0.2)))
        self.label_menu['sign_americano'].setMovie(self.movie_menu['americano'])
        self.label_menu['sign_cafelatte'].setMovie(self.movie_menu['cafelatte'])
        self.label_menu['sign_cafemocha'].setMovie(self.movie_menu['cafemocha'])
        self.label_menu['sign_americano'].setAlignment(QtCore.Qt.AlignCenter)
        self.label_menu['sign_cafelatte'].setAlignment(QtCore.Qt.AlignCenter)
        self.label_menu['sign_cafemocha'].setAlignment(QtCore.Qt.AlignCenter)
        self.label_menu['text_americano'].setFont(self.font_small)
        self.label_menu['text_cafelatte'].setFont(self.font_small)
        self.label_menu['text_cafemocha'].setFont(self.font_small)
        self.label_menu['text_americano'].setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        self.label_menu['text_americano'].setStyleSheet("margin-top: 10px")
        self.label_menu['text_cafelatte'].setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        self.label_menu['text_cafelatte'].setStyleSheet("margin-top: 10px")
        self.label_menu['text_cafemocha'].setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        self.label_menu['text_cafemocha'].setStyleSheet("margin-top: 10px")

        self.vbox_menu['americano'].addWidget(self.label_menu['sign_americano'])
        self.vbox_menu['americano'].addWidget(self.label_menu['text_americano'])
        self.vbox_menu['cafelatte'].addWidget(self.label_menu['sign_cafelatte'])
        self.vbox_menu['cafelatte'].addWidget(self.label_menu['text_cafelatte'])
        self.vbox_menu['cafemocha'].addWidget(self.label_menu['sign_cafemocha'])
        self.vbox_menu['cafemocha'].addWidget(self.label_menu['text_cafemocha'])

        self.frame_menu['americano'].setLayout(self.vbox_menu['americano'])
        self.frame_menu['americano'].setProperty('class', 'menu')
        self.frame_menu['cafelatte'].setLayout(self.vbox_menu['cafelatte'])
        self.frame_menu['cafelatte'].setProperty('class', 'menu')
        self.frame_menu['cafemocha'].setLayout(self.vbox_menu['cafemocha'])
        self.frame_menu['cafemocha'].setProperty('class', 'menu')

        self.hbox_menu.addWidget(self.frame_menu['americano'])
        self.hbox_menu.addWidget(self.frame_menu['cafelatte'])
        self.hbox_menu.addWidget(self.frame_menu['cafemocha'])

        self.set_order_visible(False)

        # 옵션 화면
        self.label_option_ment = QtWidgets.QLabel('화면에 나온 수어를 참고해 옵션을 선택해주세요')
        self.label_option_ment.setAlignment(QtCore.Qt.AlignCenter)
        self.label_option_ment.setFont(self.font_medium)
        self.label_option_ment.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.label_option_ment.setObjectName('label_ment')
        self.vbox_option = {
            'ice': QtWidgets.QVBoxLayout(),
            'hot': QtWidgets.QVBoxLayout()
        }
        self.frame_option = {
            'ice': QtWidgets.QFrame(),
            'hot': QtWidgets.QFrame()
        }
        self.hbox_option = QtWidgets.QHBoxLayout()
        self.hbox_option.setAlignment(QtCore.Qt.AlignCenter)

        self.label_option = {
            'sign_ice': QtWidgets.QLabel(),
            'sign_hot': QtWidgets.QLabel(),
            'text_ice': QtWidgets.QLabel('아이스'),
            'text_hot': QtWidgets.QLabel('핫')
        }
        self.movie_option = {
            'ice': QtGui.QMovie('data/아이스.gif'),
            'hot': QtGui.QMovie('data/핫.gif')
        }
        self.movie_option['ice'].setScaledSize(
            QtCore.QSize(int(self.screen_size.height() * 0.2), int(self.screen_size.height() * 0.2)))
        self.movie_option['hot'].setScaledSize(
            QtCore.QSize(int(self.screen_size.height() * 0.2), int(self.screen_size.height() * 0.2)))
        self.label_option['sign_ice'].setMovie(self.movie_option['ice'])
        self.label_option['sign_hot'].setMovie(self.movie_option['hot'])
        self.label_option['sign_ice'].setAlignment(QtCore.Qt.AlignCenter)
        self.label_option['sign_hot'].setAlignment(QtCore.Qt.AlignCenter)
        self.label_option['text_ice'].setFont(self.font_small)
        self.label_option['text_hot'].setFont(self.font_small)
        self.label_option['text_ice'].setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        self.label_option['text_hot'].setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)

        self.vbox_option['ice'].addWidget(self.label_option['sign_ice'])
        self.vbox_option['ice'].addWidget(self.label_option['text_ice'])
        self.vbox_option['hot'].addWidget(self.label_option['sign_hot'])
        self.vbox_option['hot'].addWidget(self.label_option['text_hot'])

        self.frame_option['ice'].setLayout(self.vbox_option['ice'])
        self.frame_option['ice'].setProperty('class', 'option')
        self.frame_option['hot'].setLayout(self.vbox_option['hot'])
        self.frame_option['hot'].setProperty('class', 'option')

        self.hbox_option.addWidget(self.frame_option['ice'])
        self.hbox_option.addWidget(self.frame_option['hot'])

        self.set_option_visible(False)

        # 주문 확인 화면
        self.label_ordered_menu = QtWidgets.QLabel()
        self.label_ordered_menu.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ordered_menu.setFont(self.font_big)
        self.label_ordered_menu.setObjectName('label_order')
        self.label_check = QtWidgets.QLabel('주문내용이 맞으신가요?')
        self.label_check.setAlignment(QtCore.Qt.AlignCenter)
        self.label_check.setFont(self.font_big)
        self.label_check.setStyleSheet("color: #065242")

        self.btn_check_n = QtWidgets.QPushButton('다시하기')
        self.btn_check_n.clicked.connect(lambda: self.reorder(current_state))
        self.btn_check_n.setFont(self.font_small)
        self.btn_check_n.setObjectName('btn_check')
        self.btn_check_next = QtWidgets.QPushButton('다음으로')
        self.btn_check_next.clicked.connect(self.go_next)
        self.btn_check_next.setFont(self.font_small)
        self.btn_check_next.setObjectName('btn_check')
        self.btn_check_add = QtWidgets.QPushButton('추가주문')
        self.btn_check_add.clicked.connect(self.add_order)
        self.btn_check_add.setFont(self.font_small)
        self.btn_check_add.setObjectName('btn_check')
        self.btn_check_y = QtWidgets.QPushButton('주문완료')
        self.btn_check_y.clicked.connect(self.finish)
        self.btn_check_y.setFont(self.font_small)
        self.btn_check_y.setObjectName('btn_check')
        self.hbox_check = QtWidgets.QHBoxLayout()
        self.hbox_check.addWidget(self.btn_check_n)
        self.hbox_check.addWidget(self.btn_check_next)
        self.hbox_check.addWidget(self.btn_check_add)
        self.hbox_check.addWidget(self.btn_check_y)
        self.btn_home = QtWidgets.QPushButton('처음으로')
        self.btn_home.clicked.connect(self.restart)
        self.btn_home.setFont(self.font_small)
        self.btn_home.setObjectName('btn_check')

        self.set_check_visible(False)

        # 수어 인식 실패 화면
        self.label_failed = QtWidgets.QLabel('수어인식에 실패하였습니다.')
        self.label_failed.setAlignment(QtCore.Qt.AlignCenter)
        self.label_failed.setFont(self.font_big)
        self.label_failed.setObjectName('label_order')
        
        self.set_failed_visible(False)

        # 최종 주문 확인 화면
        self.label_final_order = QtWidgets.QLabel()
        self.label_final_order.setAlignment(QtCore.Qt.AlignCenter)
        self.label_final_order.setFont(self.font_big)
        self.label_final_order.setObjectName('label_order')

        self.set_final_order_visible(False)

        # 수량 선택 화면
        self.label_quantity_ment = QtWidgets.QLabel('손가락으로 음료 수량을 선택해주세요.')
        self.label_quantity_ment.setAlignment(QtCore.Qt.AlignCenter)
        self.label_quantity_ment.setFont(self.font_medium)
        self.label_quantity_ment.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.label_quantity_ment.setObjectName('label_ment')

        self.set_count_visible(False)

        # 전체 레이아웃
        self.vbox = QtWidgets.QVBoxLayout()
        # self.vbox.setAlignment(QtCore.Qt.AlignHCenter)
        self.vbox.addWidget(self.label_welcome)
        self.vbox.addWidget(self.btn_start)
        self.vbox.addWidget(self.label_menu_ment)
        self.vbox.addWidget(self.label_option_ment)
        self.vbox.addWidget(self.label_quantity_ment)
        self.vbox.addWidget(self.label_cam)
        self.vbox.addLayout(self.hbox_menu)
        self.vbox.addLayout(self.hbox_option)
        self.vbox.addWidget(self.label_ordered_menu)
        self.vbox.addWidget(self.label_check)
        self.vbox.addWidget(self.label_failed)
        self.vbox.addLayout(self.hbox_check)
        self.vbox.addWidget(self.label_final_order)
        self.vbox.addWidget(self.btn_home)

        self.setLayout(self.vbox)

    # 시작 화면 표시 여부 설정
    def set_main_visible(self, visible):
        if visible:
            self.theme = 'dark'
        else:
            self.theme = 'light'
        self.label_welcome.setVisible(visible)
        self.btn_start.setVisible(visible)

    # 주문 화면 표시 여부 설정
    def set_order_visible(self, visible):
        self.label_menu_ment.setVisible(visible)
        self.label_cam.setVisible(visible)
        for k, v in self.frame_menu.items():
            v.setVisible(visible)
        for k, v in self.movie_menu.items():
            if visible: v.start()
            else: v.stop()

    # 옵션 화면 표시 여부 설정
    def set_option_visible(self, visible):
        self.label_option_ment.setVisible(visible)
        self.label_cam.setVisible(visible)
        for k, v in self.frame_option.items():
            v.setVisible(visible)
        for k, v in self.movie_option.items():
            if visible: v.start()
            else: v.stop()

    def set_count_visible(self, visible):
        self.label_cam.setVisible(visible)
        self.label_quantity_ment.setVisible(visible)

    # 주문 확인 화면 표시 여부 설정
    def set_check_visible(self, visible):
        global current_state
        self.label_ordered_menu.setVisible(visible)
        self.label_check.setVisible(visible)
        self.btn_check_n.setVisible(visible)
        if current_state in (state.ORDER_MENU, state.ORDER_OPTION):
            self.btn_check_next.setVisible(visible)
        elif current_state == state.ORDER_COUNT:
            self.btn_check_add.setVisible(visible)
            self.btn_check_y.setVisible(visible)
        else:
            self.btn_check_next.setVisible(visible)
            self.btn_check_add.setVisible(visible)
            self.btn_check_y.setVisible(visible)
        self.btn_home.setVisible(visible)

    # 수어 인식 실패 화면 표시 여부 설정
    def set_failed_visible(self, visible):
        self.label_failed.setVisible(visible)
        self.btn_check_n.setVisible(visible)
        self.btn_home.setVisible(visible)

    # 최종 주문 확인 화면 표시 여부
    def set_final_order_visible(self, visible):
        self.label_final_order.setVisible(visible)
        self.btn_home.setVisible(visible)

    # 주문 시작: CamThread
    def start(self):
        global running, current_state
        current_state = state.ORDER_MENU
        if not running:
            running = True
            self.thread_cam.start()
            self.set_main_visible(False)
            self.set_order_visible(True)
            self.movie_menu['americano'].start()
            self.movie_menu['cafelatte'].start()
            self.movie_menu['cafemocha'].start()
            print('{}, {}'.format(self.width(), self.height()))
            print('started..')
        else:
            self.set_main_visible(False)
            self.reorder()

    # 다시 주문
    def reorder(self, st=state.ORDER_MENU):
        global start_capture, sentence, order_list, current_state, gesture_displayed, gesture_counts
        start_capture = True
        sentence = []
        self.set_check_visible(False)
        self.set_failed_visible(False)
        self.set_final_order_visible(False)
        if st == state.ORDER_MENU:
            self.set_order_visible(True)
        elif st == state.ORDER_OPTION:
            self.set_option_visible(True)
        elif st == state.ORDER_COUNT:
            gesture_displayed = False
            gesture_counts = {i: 0 for i in range(len(gesture))}
            self.set_count_visible(True)
        current_state = st

    def go_next(self):
        global current_state, gesture_displayed, gesture_counts
        if current_state == state.ORDER_MENU:
            # todo: go to order option
            self.reorder(state.ORDER_OPTION)
        elif current_state == state.ORDER_OPTION:
            # todo: go to order count
            gesture_displayed = False
            gesture_counts = {i: 0 for i in range(len(gesture))}
            self.reorder(state.ORDER_COUNT)

    # 주문 추가
    def add_order(self):
        global current_state, order_list
        for d in order_list:
            if self.last_order['menu'] == d['menu'] and self.last_order['option'] == d['option']:
                d['count'] = d['count'] + self.last_order['count']
                break
        else:
            order_list.append(self.last_order)
        self.reorder()

    # 주문 종료
    def finish(self):
        global running, order_list, current_state
        if not self.last_order == dict():
            for d in order_list:
                if self.last_order['menu'] == d['menu'] and self.last_order['option'] == d['option']:
                    d['count'] = d['count'] + self.last_order['count']
                    break
            else:
                order_list.append(self.last_order)
        print(order_list)

        result = '\n'.join([f"{d['menu']} ({d['option']}) - {d['count']}잔" for d in order_list])
        print(result)
        self.set_check_visible(False)
        self.set_failed_visible(False)
        self.label_final_order.setText(result + '\n\n 주문이 완료되었습니다.')
        self.set_final_order_visible(True)
        current_state = state.ORDER_NONE

    # 처음부터
    def restart(self):
        global current_state, order_list
        order_list = []
        for k, v in self.movie_menu.items():
            v.stop()
        for k, v in self.movie_option.items():
            v.stop()
        self.set_check_visible(False)
        self.set_failed_visible(False)
        self.set_final_order_visible(False)
        self.set_main_visible(True)
        current_state = state.ORDER_NONE

    @QtCore.pyqtSlot(str)
    def set_ordered_menu(self, _order):
        global current_state
        print('sign language detection: ' + _order)
        if current_state == state.ORDER_MENU and _order in actions_ko['menu']:
            self.last_order = dict()
            self.last_order['menu'] = actions_ko['menu'][_order]
            self.label_ordered_menu.setText(self.last_order['menu'])
        elif current_state == state.ORDER_OPTION and _order in actions_ko['option']:
            self.last_order['option'] = actions_ko['option'][_order]
            self.label_ordered_menu.setText(self.last_order['option'])
        elif current_state == state.ORDER_COUNT and _order in ('1', '2', '3', '4', '5'):
            self.last_order['count'] = int(_order)
            self.label_ordered_menu.setText(str(self.last_order['count']) + '잔')

    @pyqtProperty(str)
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, theme):
        # Register change of state
        self._theme = theme
        # Update displayed style
        self.style().polish(self)

app = QtWidgets.QApplication([])
widget = MainWidget()

# desktop = QDesktopWidget()
# screen_rect = desktop.availableGeometry(desktop.primaryScreen())
# widget.setGeometry(0, 0, screen_rect.width(), screen_rect.height()-20)

widget.showMaximized()
sys.exit(app.exec_())
