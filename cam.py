import cv2
import numpy as np
import tensorflow as tf

# 저장된 모델 불러오기
my_model = tf.keras.models.load_model('stair4.h5')

# 카메라 캡처q
cam = cv2.VideoCapture(cv2.CAP_DSHOW+0)

while True:
    ret, frame = cam.read()  # 프레임 읽기
    if not ret:
        break

    # 프레임 크기 조정 및 색상 채널 변경
    resized_frame = cv2.resize(frame, (64, 64))
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # 모델 입력 형식에 맞게 배열 형태로 변환
    input_image = np.expand_dims(resized_frame, axis=0)
    
    # 모델 예측
    predictions = my_model.predict(input_image)

    # 클래스 확인
    class_index = np.argmax(predictions)

    # 클래스에 따른 표시 문자열 생성
    if class_index == 0:
        label = "STAIR"
    elif class_index == 1:
        label = "CURB"
    else:
        label = "ROAD"

    # 이미지에 예측된 클래스 표시
    cv2.putText(frame, label, (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면에 프레임 표시
    cv2.imshow('Classification', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 및 창 해제
cam.release()
cv2.destroyAllWindows()