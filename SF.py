import os
import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def slice_yolo_refined(img_path, lbl_path, output_img_dir, output_lbl_dir, slice_size=640, overlap=0.5, bg_ratio=0.2):
    """
    고해상도 이미지를 슬라이싱하고 가시성(Visibility)을 체크하여 라벨을 정제하는 함수
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    
    h, w, _ = img.shape
    
    # 원본 라벨 읽기
    labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
    
    # 슬라이싱 간격 (Overlap 적용)
    step = int(slice_size * (1 - overlap))
    patch_count = 0
    
    # 슬라이싱 루프
    for y in range(0, max(1, h - slice_size + 1), step):
        for x in range(0, max(1, w - slice_size + 1), step):
            y1, x1 = y, x
            y2, x2 = min(h, y + slice_size), min(w, x + slice_size)
            
            # 이미지 경계면 보정
            if y2 - y1 < slice_size: y1 = max(0, y2 - slice_size)
            if x2 - x1 < slice_size: x1 = max(0, x2 - slice_size)

            slice_img = img[y1:y2, x1:x2]
            new_labels = []
            
            for lbl in labels:
                cls = int(float(lbl[0]))
                x_cn, y_cn, w_n, h_n = map(float, lbl[1:])
                
                # 1. 원본 기준 절대 좌표 변환
                abs_w, abs_h = w_n * w, h_n * h
                abs_cx, abs_cy = x_cn * w, y_cn * h
                xmin, ymin = abs_cx - abs_w / 2, abs_cy - abs_h / 2
                xmax, ymax = abs_cx + abs_w / 2, abs_cy + abs_h / 2
                
                # 2. 현재 조각과 객체의 겹치는 영역(Intersection) 계산
                inter_xmin, inter_ymin = max(x1, xmin), max(y1, ymin)
                inter_xmax, inter_ymax = min(x2, xmax), min(y2, ymax)
                
                if inter_xmax > inter_xmin and inter_ymax > inter_ymin:
                    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
                    orig_area = abs_w * abs_h
                    visibility = inter_area / orig_area # 가시성 비율 (0~1)
                    
                    # 3. [핵심] 품질 필터링: 논문 기반의 유효성 검사
                    is_valid = False
                    if cls == 1:  # Worker 클래스
                        # 사람이 원본 대비 40% 이상 보이고, 최소 크기가 40px 이상일 때
                        if visibility > 0.4 and (inter_xmax - inter_xmin) > 40:
                            is_valid = True
                    elif cls == 0:  # Hook_Fastened 클래스
                        # 후크는 작으므로 더 엄격하게 (60% 이상 보일 때만)
                        if visibility > 0.6 and (inter_xmax - inter_xmin) > 10:
                            is_valid = True
                    
                    if is_valid:
                        # 4. 조각(640) 기준 상대 좌표로 재변환
                        new_cx = (inter_xmin + (inter_xmax - inter_xmin)/2 - x1) / slice_size
                        new_cy = (inter_ymin + (inter_ymax - inter_ymin)/2 - y1) / slice_size
                        new_w = (inter_xmax - inter_xmin) / slice_size
                        new_h = (inter_ymax - inter_ymin) / slice_size
                        
                        new_labels.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")
            
            # 5. 데이터 저장 결정 (객체가 있거나, 설정된 확률로 배경 타일 저장)
            if new_labels or random.random() < bg_ratio:
                base_name = f"{img_path.stem}_p{patch_count}"
                img_out_path = os.path.join(output_img_dir, f"{base_name}.png")
                lbl_out_path = os.path.join(output_lbl_dir, f"{base_name}.txt")
                
                cv2.imwrite(img_out_path, slice_img)
                with open(lbl_out_path, 'w') as f:
                    f.write('\n'.join(new_labels))
                patch_count += 1
                
    return patch_count

def main():
    # --- 설정 영역 ---
    INPUT_ROOT = "merged_dataset"       # 원본 데이터 폴더
    OUTPUT_ROOT = "final_dataset_sahi" # 결과 저장 폴더
    SLICE_SIZE = 640
    OVERLAP = 0.5
    BG_RATIO = 0.2  # 배경 오탐을 줄이기 위해 배경 타일 20% 포함
    # ----------------

    for split in ['train', 'val']:
        print(f"\nProcessing {split} set...")
        img_in_dir = Path(INPUT_ROOT) / split / 'images'
        lbl_in_dir = Path(INPUT_ROOT) / split / 'labels'
        
        img_out_dir = Path(OUTPUT_ROOT) / split / 'images'
        lbl_out_dir = Path(OUTPUT_ROOT) / split / 'labels'
        
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        
        img_files = list(img_in_dir.glob("*.jpg")) + list(img_in_dir.glob("*.png"))
        total_patches = 0
        
        for img_path in tqdm(img_files):
            lbl_path = lbl_in_dir / (img_path.stem + ".txt")
            total_patches += slice_yolo_refined(
                img_path, lbl_path, img_out_dir, lbl_out_dir, 
                SLICE_SIZE, OVERLAP, BG_RATIO
            )
            
        print(f"Finished {split}: Generated {total_patches} patches.")

    print("\n✅ 모든 프로세스가 완료되었습니다! 'final_dataset_sahi' 폴더를 확인하세요.")

if __name__ == "__main__":
    main()