#!/usr/bin/env python3
import cv2
from ultralytics import YOLO

def main():
    # 1. 학습한 가중치 파일(.pt) 경로 지정
    model_path = "/home/rokey/runs/detect/runs/yolov10n-7/weights/best_arg_epoch100.pt" 
    
    print(f"🔄 YOLO 트래킹 모델 로드 중: {model_path}")
    model = YOLO(model_path)
    print("✅ 모델 로드 완료!")

    # 2. USB 카메라 연결 (기존 설정하신 2번 인덱스 유지)
    cap = cv2.VideoCapture(2)

    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 에러: USB 카메라를 열 수 없습니다. 인덱스(2)를 확인하세요.")
        return

    print("🚀 실시간 트래킹(Tracking)을 시작합니다. 종료하려면 'q'를 누르세요.")

    while True:
        # 카메라로부터 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 가져올 수 없습니다. 스트림을 확인하세요.")
            break

        # 3. model() 대신 model.track() 사용 
        # - persist=True: 이전 프레임의 추적 ID 정보가 다음 프레임으로 계속 유지(연장)되도록 합니다.
        # - stream=True: 비디오 스트림 처리 시 메모리 축적을 방지합니다.
        results = model.track(frame, persist=True, stream=True)

        annotated_frame = frame.copy()

        # 4. 추득된 결과에서 바운딩 박스와 추적 ID 입히기
        for r in results:
            # 기본 바운딩 박스 및 클래스명 시각화화
            annotated_frame = r.plot()
            
            # (옵션) 콘솔이나 로직 제어단에서 개별 추적 ID 번호(정수형)가 필요할 때 참조하는 방법
            # 로봇 제어나 픽앤플레이스 좌표 지정 시 r.boxes.id 값을 활용하게 됩니다.
            if r.boxes is not None and r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
                # print(f"현재 감지된 물체들의 고유 ID 라인업: {track_ids}")

        # 5. 트래킹 결과 화면 출력
        cv2.imshow("YOLOv10 Real-Time Tracking", annotated_frame)

        # 'q' 키를 누르면 루프 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("👋 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()