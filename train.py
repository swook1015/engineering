from ultralytics import YOLO

def train_hybrid():
    model = YOLO("yolo26m.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,

        optimizer='SGD',
        lr0=0.001,
        momentum=0.9,
        weight_decay=0.0001,
        warmup_epochs=3,

        batch=8,          # VRAM 여유가 있다면 16으로 올려도 좋습니다.
        device=0,
        workers=0,

        # --- 데이터 증강(Augmentation) 수정 ---
        mosaic=0.5,       # 조각난 데이터이므로 강도 하향 (1.0 -> 0.5)
        close_mosaic=10,  # 마지막 10에폭은 모자이크 없이 현실 데이터 적응 (유지)
        copy_paste=0.0,   # 경계면 부자연스러움 방지 (0.1 -> 0.0)
        flipud=0.0,       # 상하 반전 금지 (0.5 -> 0.0)
        fliplr=0.5,       # 좌우 반전은 유지
        degrees=10.0,     # 과도한 회전 방지 (15.0 -> 10.0)
        translate=0.1,
        mixup=0.0,        # 노이즈 방지를 위해 비활성화 (0.2 -> 0.0)
        # ------------------------------------

        box=10.0,
        cls=2.5,

        project='C:/Users/pcroom2/Desktop/engineering/runs',
        name='yolo26m_hybrid_worker_fix',
        exist_ok=True
    )

    metrics = model.val()
    print(f"\n=== 최종 검증 결과 ===")
    print(f"전체  mAP50: {metrics.box.map50:.4f}")
    print(f"클래스별: {metrics.box.maps}")

if __name__ == "__main__":
    train_hybrid()