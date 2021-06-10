import cv2   # OpenCV
import mediapipe as mp   # Mediapipe
import numpy as np   # 수치 연산을 위한 Numpy 사용

mp_drawing = mp.solutions.drawing_utils   # 포즈 랜드마크를 그릴 수 있게 도와주는 유틸리티
mp_pose = mp.solutions.pose

### 각도 계산 함수 ###
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])   # 세 점의 각도(radian)를 np.arctan2(y,x)를 이용해 계산
    angle = np.abs(radians*180.0/np.pi)   # angle 계산
    
    if angle >180.0:
        angle = 360-angle   # 각도의 최대값 180도
        
    return angle   # angle 값 반환


### Mediapipe Solution ###
cap = cv2.VideoCapture(0)   # OpenCV의 VideoCapture를 이용해 웹캠 사용
 
squat_count = 0   # 카운트 초기화
squat_state = None   # 상태 초기화

with mp_pose.Pose(   
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

  while cap.isOpened():    # 캠이 열려 있으면 반복
    success, image = cap.read()   # 성공하면 아래 코드를 읽음
    if not success:   # 실패하면 다음 프레임 진행
      continue

    
    # 카메라를 수평으로 뒤집고(거울모드) 색상 코드 BGR을 RGB로 변환(opencv는 BGR 코드 색상, mediapipe는 RGB 코드 색상 사용)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = pose.process(image)   # 전처리 및 모델 추론 실행

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   # 다시 RGB를 BGR로 변환

    try:
        landmarks = results.pose_landmarks.landmark   # 랜드마크 좌표 추출
            
        ### 각각의 좌표 저장 ### (거울모드이므로 반대로 저장)

        # 다리
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]   # 힙 좌표(x,y) 저장
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]   # 무릎 좌표(x,y) 저장
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]   # 발목 좌표(x,y) 저장

        ### 각도 계산 ###
        leg_angle = calculate_angle(hip, knee, ankle)   # 힙, 무릎, 발목의 각도 계산

        ### 각도 커스텀 출력 ###
        # 다리
        cv2.putText(image, str(leg_angle),
            tuple(np.multiply(knee, [640, 480]).astype(int)),   # 무릎 부분에 Text 출력
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )

        # 스쿼트 카운트
        if leg_angle < 140:   # 각도가 140보다 작으면(다리를 굽히면),
            squat_state = "down"   # down
        if leg_angle > 160 and squat_state == 'down':   # 다리 각도가 170보다 크고, state가 down이면(다리을 펴면)
            squat_state = "up"   # up
            squat_count += 1   # count + 1
            print(squat_count)
            
    except:
        pass

    cv2.putText(image, 'Squat', (490,440), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # 스쿼트 카운트 출력
    cv2.putText(image, str(squat_count), 
                (490,468), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        
    # 스쿼트 상태 출력
    cv2.putText(image, squat_state, 
                (540,468), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

    ### 랜드마크 드로잉 ###
    mp_drawing.draw_landmarks(   
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    ### 웹캠 출력 ###
    cv2.imshow('Exercise Count', image)   # OpenCV imshow를 이용해 결과물 출력
    if cv2.waitKey(5) & 0xFF == ord('q'):   # q 누르면 웹캠 종료
        break

cap.release()