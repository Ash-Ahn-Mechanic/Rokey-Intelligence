#!/usr/bin/env python3
import cv2
from ultralytics import YOLO

def main():
    # 1. 학습한 가중치 파일(.pt) 경로 지정
    # 예: 'yolov10n.pt' 또는 프로젝트 폴더 내 'best.pt' 경로
    model_path = "/home/rokey/runs/detect/runs/yolov10n-7/weights/best_arg_epoch100.pt" 
    
    print(f"🔄 YOLO 모델 로드 중: {model_path}")
    model = YOLO(model_path)
    print("✅ 모델 로드 완료!")

    # 2. USB 카메라 연결 (기본 웹캠은 보통 0번, 외장 캠은 1번이나 2번)
    cap = cv2.VideoCapture(2)

    # 카메라 해상도 설정 (원하는 해상도로 조절 가능, 기본값으로 두려면 주석 처리)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 에러: USB 카메라를 열 수 없습니다. 인덱스(0)를 확인하세요.")
        return

    print("🚀 실시간 추론을 시작합니다. 종료하려면 'q'를 누르세요.")

    while True:
        # 카메라로부터 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 가져올 수 없습니다. 스트림을 확인하세요.")
            break

        # 3. YOLO 모델로 실시간 추론 진행
        # stream=True 옵션은 실시간 영상 처리 시 메모리 효율을 극대화합니다.
        results = model(frame, stream=True)

        # 4. 추론 결과를 프레임 위에 그리기
        # ultralytics에서 제공하는 .plot() 기능을 쓰면 바운딩 박스와 클래스명이 자동으로 입혀집니다.
        for r in results:
            annotated_frame = r.plot()

        # 5. 화면에 출력
        cv2.imshow("YOLOv10 Real-Time Detection", annotated_frame)

        # 'q' 키를 누르면 루프 탈출 (종료)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("👋 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()