import mediapipe as mp
import numpy as np
import cv2

mp_holistic = mp.solutions.holistic # Holistic model (face, pose, left/right hand 인식 가능 모듈)
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

class Frame():
    def __init__(self, image, colors=[(245,117,16), (117,245,16), (16,117,245), (255,20,147), (255,20,147)]):
        # Mediapipe 설정
        self.image = image
        self.colors = colors

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    # Mediapipe 관절 인식 함수
    def mediapipe_detection(self, model):
        image = cv2.flip(self.image,1)                      # 이미지 좌/우 반전
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB (OpenCV 영상은 BRG 형식, Mediapipe는 RGB 형식이기 때문에)
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction (result에 detection한 결과 값을 저장)RGB로 변환했던 것을 OpenCV 영상처리를 위해 다시 BRG로 되돌림)
        return results

    # 랜드마크 그리기 함수
    def draw_landmarks(self, results):
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # 얼굴 랜드마크
        mp_drawing.draw_landmarks(self.image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(self.image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(self.image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # 랜드마크 스타일 커스텀 그리기 함수
    def draw_styled_landmarks(self, results):
        # Draw face connections 얼굴 랜드마크
        """"mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )"""
        # Draw pose connections
        mp_drawing.draw_landmarks(self.image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
        # Draw left hand connections
        mp_drawing.draw_landmarks(self.image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
        # Draw right hand connections
        mp_drawing.draw_landmarks(self.image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

    # 랜드마크 좌표 추출 함수
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404) 얼굴 랜드마크
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])

    # 실시간 감지 화면 설정
    def prob_viz(self, res, actions, input_frame):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), self.colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

        return output_frame