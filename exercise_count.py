import cv2   # OpenCV
import mediapipe as mp   # Mediapipe

mp_drawing = mp.solutions.drawing_utils   # 포즈 랜드마크를 그릴 수 있게 도와주는 유틸리티
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)   # OpenCV의 VideoCapture를 이용해 웹캠 사용
with mp_pose.Pose(   
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

  while cap.isOpened():    # 캠이 열려 있으면 반복
    success, image = cap.read()   # 성공하면 아래 코드를 읽음
    if not success:   # 실패하면 다음 프레임 진행
      continue

    # 카메라를 수평으로 뒤집고 색상 코드 BGR을 RGB로 변환(opencv는 BGR 코드 색상, mediapipe는 RGB 코드 색상 사용)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = pose.process(image)   # 전처리 및 모델 추론 실행

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   # 다시 RGB를 BGR로 변환

    mp_drawing.draw_landmarks(   # 랜드마크 드로잉
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Exercise Count', image)   # OpenCV imshow를 이용해 결과물 출력
    if cv2.waitKey(5) & 0xFF == ord('q'):   # q 누르면 웹캠 종료
      break
cap.release()