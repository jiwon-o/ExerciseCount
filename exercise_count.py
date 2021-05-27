import cv2   # OpenCV
import mediapipe as mp   # Mediapipe
import numpy as np   # 수치 연산을 위한 Numpy 사용

mp_drawing = mp.solutions.drawing_utils   # 포즈 랜드마크를 그릴 수 있게 도와주는 유틸리티
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])   # 세 점의 각도(radian)를 np.arctan2(y,x)를 이용해 계산
    angle = np.abs(radians*180.0/np.pi)   # angle 계산
    
    if angle >180.0:
        angle = 360-angle   # 각도의 최대값 180도
        
    return angle   # angle 값 반환

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


    try:
        landmarks = results.pose_landmarks.landmark   # 랜드마크 좌표 추출
            
        ### 각각의 좌표 저장 ###
        # 왼쪽 팔
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]   # 왼쪽 어깨 좌표(x,y) 저장
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]   # 왼쪽 팔꿈치 좌표(x,y) 저장
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]   # 왼쪽 손목 좌표(x,y) 저장
        
        # 오른쪽 팔
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]   # 오른쪽 어깨 좌표(x,y) 저장
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]   # 오른쪽 팔꿈치 좌표(x,y) 저장
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]   # 오른쪽 손목 좌표(x,y) 저장

        # 왼쪽 다리
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]   # 왼쪽 힙 좌표(x,y) 저장
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]   # 왼쪽 무릎 좌표(x,y) 저장
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]   # 왼쪽 발목 좌표(x,y) 저장


        ### 각도 계산 ###
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)   # 왼쪽 어깨, 팔꿈치, 손목의 각도 계산
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)   # 오른쪽 어깨, 팔꿈치, 손목의 각도 계산
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)   # 왼쪽 힙, 무릎, 발목의 각도 계산

        ### 각도 커스텀 출력 ###
        # 왼쪽 팔
        cv2.putText(image, str(left_arm_angle),   
            tuple(np.multiply(left_elbow, [640, 480]).astype(int)),   # 팔꿈치 부분에 Text 출력
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )

        # 오른쪽 팔
        cv2.putText(image, str(right_arm_angle),   
            tuple(np.multiply(right_elbow, [640, 480]).astype(int)),   # 팔꿈치 부분에 Text 출력
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )

        # 왼쪽 다리
        cv2.putText(image, str(left_leg_angle),
            tuple(np.multiply(left_knee, [640, 480]).astype(int)),   # 무릎 부분에 Text 출력
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )
                       
    except:
        pass

    mp_drawing.draw_landmarks(   # 랜드마크 드로잉
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Exercise Count', image)   # OpenCV imshow를 이용해 결과물 출력
    if cv2.waitKey(5) & 0xFF == ord('q'):   # q 누르면 웹캠 종료
        break
cap.release()

