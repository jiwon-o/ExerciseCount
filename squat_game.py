import cv2   # OpenCV
import mediapipe as mp   # Mediapipe
import numpy as np   # 수치 연산을 위한 Numpy 사용
import pygame

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


# 게임 초기화
pygame.init()

# 게임창 옵션 설정
size = [1920, 1080]
screen = pygame.display.set_mode(size)
title = "Squat Game"
pygame.display.set_caption(title)

# 게임 내 필요한 설정
clock = pygame.time.Clock()
black = (0, 0, 0)
white = (255, 255, 255)

# pygame 이미지 로드
hi = pygame.image.load('images/img_hi.PNG').convert_alpha()
hi = pygame.transform.scale(hi, (1900, 1060))
ok = pygame.image.load('images/img_ok.PNG').convert_alpha()
ok = pygame.transform.scale(ok, (1900, 1060))
ready = pygame.image.load('images/img_ready.PNG').convert_alpha()
ready = pygame.transform.scale(ready, (1900, 1060))
start = pygame.image.load('images/img_start.PNG').convert_alpha()
start = pygame.transform.scale(start, (1900, 1060))
up = pygame.image.load('images/img_up.PNG').convert_alpha()
up = pygame.transform.scale(up, (1900, 1060))
down = pygame.image.load('images/img_down.PNG').convert_alpha()
down = pygame.transform.scale(down, (1900, 1060))

### Mediapipe Solution ###
cap = cv2.VideoCapture(0)   # OpenCV의 VideoCapture를 이용해 웹캠 사용

squat_count = 0   # 카운트 초기화
ready_state = None   # 상태 초기화
ok_state = None
squat_state = None

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
        # 오른팔
        right_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]   # 오른쪽 어깨 좌표(x,y) 저장
        right_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]   # 오른쪽 팔꿈치 좌표(x,y) 저장
        right_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]   # 오른쪽 손목 좌표(x,y) 저장

        # 왼팔
        left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]   # 왼쪽 어깨 좌표(x,y) 저장
        left_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]   # 왼쪽 팔꿈치 좌표(x,y) 저장
        left_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]   # 왼쪽 손목 좌표(x,y) 저장
        
        # 다리
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]   # 힙 좌표(x,y) 저장
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]   # 무릎 좌표(x,y) 저장
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]   # 발목 좌표(x,y) 저장

    except:
        pass
            
    # pygame FPS 설정
    clock.tick(60)

    # pygame 각종 입력 감지
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == 'q':   # q누르면 게임 종료
                break


    # 배경
    screen.blit(hi, (0,0))   # 처음 화면

    ### 각도 계산 ###
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)   # 왼쪽 어깨, 팔꿈치, 손목의 각도 계산
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)   # 오른쪽 어깨, 팔꿈치, 손목의 각도 계산
    leg_angle = calculate_angle(hip, knee, ankle)   # 힙, 무릎, 발목의 각도 계산

    ### 좌표 설정 ###
    diff = abs(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x - landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x)   # 오른쪽 손목과 왼쪽 손목의 거리
    dist = int(diff * 500)

    left_pinky_y = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y   # 왼쪽 새끼손가락 y좌표
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y   # 왼쪽 어깨 y좌표
    right_pinky_y = landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y   # 오른쪽 새끼손가락 y좌표
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y   # 오른쪽 어깨 y좌표

    ### 게임 시작 ###
    # 스쿼트 준비
    if left_pinky_y < left_shoulder_y or right_pinky_y < right_shoulder_y:   # 손이 어깨 위로 올라가면
        hi = ok
        screen.blit(ok, (0,0))   # 좋아요 화면
        ok_state = "ok"

    if left_pinky_y > left_shoulder_y and ok_state == 'ok':   # 손을 내리면
        ok = ready
        screen.blit(ready, (0,0))   # 스쿼트 준비 화면

    if dist < 50:   # 두 손을 모으면
        ready = start
        screen.blit(start, (0,0))   # 스쿼트 시작 화면
        ready_state = "ready"

    # 스쿼트 시작
    if leg_angle < 140 and ready_state == 'ready':   # 다리 각도가 140보다 작으면(다리를 굽히면)
        ready = up
        start = up
        screen.blit(up, (0,0))   # UP 화면
        squat_state = "down"

    if leg_angle > 170 and ready_state == 'ready' and squat_state == 'down':   # 다리 각도가 170보다 크면(다리를 펴면)
        start = down
        ready = down
        screen.blit(down, (0,0))   # DOWN 화면
        squat_state = "up"
        squat_count += 1   # 앉았다 일어나면 COUNT + 1

    # 업데이트
    pygame.display.flip()

    # 스쿼트 카운트 출력
    cv2.putText(image, str(squat_count), 
                (520,130), 
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (0,0,255), 2, cv2.LINE_AA)

    ### 랜드마크 드로잉 ###
    mp_drawing.draw_landmarks(   
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    ### 웹캠 출력 ###
    cv2.imshow('Exercise Count', image)   # OpenCV imshow를 이용해 결과물 출력
    if cv2.waitKey(5) & 0xFF == ord('q'):   # q 누르면 웹캠 종료
        break

pygame.quit()
cap.release()