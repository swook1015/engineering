import os
import cv2
import random
from pathlib import Path

def slice_yolo(img_path, lbl_path, output_img_dir, output_lbl_dir, slice_size=640, overlap=0.5, bg_ratio=0.1):
    img = cv2.imread(img_path)
    if img is None: return
    h, w, _ = img.shape
    
    # 라벨 읽기
    labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
    
    # overlap 0.5 적용: 640 조각이 320 픽셀마다 생성됨
    step = int(slice_size * (1 - overlap))
    count = 0
    
    for y in range(0, max(1, h - slice_size + 1), step):
        for x in range(0, max(1, w - slice_size + 1), step):
            # 조각의 절대 좌표 설정
            y1, x1 = y, x
            y2, x2 = min(h, y + slice_size), min(w, x + slice_size)
            
            # 해상도가 안 맞아서 남는 끝부분 보정
            if y2 - y1 < slice_size: y1 = max(0, y2 - slice_size)
            if x2 - x1 < slice_size: x1 = max(0, x2 - slice_size)

            slice_img = img[y1:y2, x1:x2]
            new_labels = []
            
            for lbl in labels:
                cls = int(float(lbl[0]))
                x_cn, y_cn, w_n, h_n = map(float, lbl[1:])
                
                # 1. 4K 기준 절대 좌표(xmin, ymin, xmax, ymax)로 변환
                abs_w, abs_h = w_n * w, h_n * h
                abs_cx, abs_cy = x_cn * w, y_cn * h
                xmin = abs_cx - abs_w / 2
                ymin = abs_cy - abs_h / 2
                xmax = abs_cx + abs_w / 2
                ymax = abs_cy + abs_h / 2
                
                # 2. 객체의 중심점이 현재 조각 안에 있는지 확인
                if x1 <= abs_cx < x2 and y1 <= abs_cy < y2:
                    # 3. 조각 경계선에 맞춰 바운딩 박스 정밀 클리핑 (삐져나감 방지)
                    inter_xmin = max(x1, xmin)
                    inter_ymin = max(y1, ymin)
                    inter_xmax = min(x2, xmax)
                    inter_ymax = min(y2, ymax)
                    
                    # 4. 640 조각 기준의 상대 좌표(0~1)로 다시 변환
                    new_xmin = (inter_xmin - x1) / slice_size
                    new_ymin = (inter_ymin - y1) / slice_size
                    new_xmax = (inter_xmax - x1) / slice_size
                    new_ymax = (inter_ymax - y1) / slice_size
                    
                    new_w = new_xmax - new_xmin
                    new_h = new_ymax - new_ymin
                    new_cx = new_xmin + new_w / 2
                    new_cy = new_ymin + new_h / 2
                    
                    # 찌꺼기 박스 필터링 (너무 작게 잘린 객체는 노이즈 방지를 위해 버림)
                    if new_w > 0.05 and new_h > 0.05:
                        new_labels.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")
            
            # 5. [중요] 객체가 있거나, 10% 확률로 빈 배경(Background)도 저장
            if new_labels or random.random() < bg_ratio:
                base_name = f"{Path(img_path).stem}_patch_{count}"
                cv2.imwrite(os.path.join(output_img_dir, f"{base_name}.png"), slice_img)
                
                # 라벨 파일 작성 (빈 리스트면 빈 텍스트 파일 생성 -> YOLO가 배경으로 인식)
                with open(os.path.join(output_lbl_dir, f"{base_name}.txt"), 'w') as f:
                    f.write('\n'.join(new_labels))
                count += 1

def process_dataset(input_base, output_base):
    for split in ['train', 'val']:
        img_dir = os.path.join(input_base, split, 'images')
        lbl_dir = os.path.join(input_base, split, 'labels')
        
        out_img_dir = os.path.join(output_base, split, 'images')
        out_lbl_dir = os.path.join(output_base, split, 'labels')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        
        if not os.path.exists(img_dir): continue
            
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]
        print(f"--- {split} 슬라이싱 시작 (총 {len(images)}장) ---")
        
        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + '.txt')
            # 라벨 파일이 없어도 배경으로 쓸 수 있도록 처리
            slice_yolo(img_path, lbl_path, out_img_dir, out_lbl_dir)
            
    print("✅ 데이터셋 재슬라이싱 완료!")

# 실행 (기존 폴더 덮어쓰지 않게 출력 폴더명을 바꿨어)
process_dataset("merged_dataset", "final_dataset_overlap50")