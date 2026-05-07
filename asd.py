from ultralytics import YOLO

def train_custom_yolo():
    # 1. 모델 로드 (YOLO26m 아키텍처 기반)
    # TPH(Transformer Prediction Head) 및 P2 레이어가 적용된 가중치 파일을 사용합니다.
    model = YOLO("yolo26m.pt") 

    # 2. 학습 설정 (논문 기반 최적 파라미터 적용)
    model.train(
        # 데이터셋 설정 파일 (절대 경로 혹은 train.py 기준 상대 경로)
        data="data.yaml",      
        
        # 에폭 및 해상도
        epochs=100,            # 충분한 수렴을 위한 설정
        imgsz=640,             # 4K에서 640x640으로 슬라이싱된 패치 크기 일치
        
        # 하이퍼파라미터 (SAHI 논문 및 항공 영상 서베이 논문 권장값)
        optimizer='SGD',       # AdamW보다 일반화 성능이 우수한 SGD 선택
        lr0=0.01,              # 초기 학습률
        momentum=0.9,          # 관성 계수
        weight_decay=0.0001,   # 가중치 감쇠 (L2 정규화)
        
        # 배치 및 장치 설정
        batch=16,              # GPU 메모리에 따라 -1(Auto) 또는 16, 32 설정
        device=0,              # GPU 번호
        
        # 성능 향상을 위한 특수 기능
        rect=False,            # 슬라이싱된 정방형 패치 학습이므로 False 권장
        mosaic=1.0,            # 초소형 객체 탐지를 위한 모자이크 증강 활성화
        close_mosaic=10,       # 학습 막바지 10에폭 동안 모자이크 해제 (정밀도 향상)
        
        # 결과 저장 설정
        project='Construction_Safety',
        name='yolo26m_hook_sahi_sf',
        exist_ok=True
    )

    # 3. 학습 완료 후 검증
    metrics = model.val()
    print(f"검증 결과 (mAP50): {metrics.box.map50}")

if __name__ == "__main__":
    train_custom_yolo()