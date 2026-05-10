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

    batch=8,          # 8 → 16 (VRAM 여유 있으면 속도 향상)
    device=0,
    workers=0,

    mosaic=1.0,
    close_mosaic=10,
    copy_paste=0.1,
    flipud=0.5,
    fliplr=0.5,
    degrees=15.0,
    translate=0.1,
    mixup=0.2,

    box=10.0,
    cls=2.5,
    # multi_scale=True  ← 제거

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