import os
import random
import shutil
from pathlib import Path

def split_dataset(source_base_dir, output_base_dir, split_ratio=0.8):
    # 1. 생성될 폴더 구조 설정
    dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for d in dirs:
        os.makedirs(os.path.join(output_base_dir, d), exist_ok=True)

    # 2. 클래스 폴더(hooked, worker) 목록 가져오기
    classes = [d for d in os.listdir(source_base_dir) if os.path.isdir(os.path.join(source_base_dir, d))]
    
    for cls in classes:
        cls_path = os.path.join(source_base_dir, cls)
        # 이미지 파일들만 추출 (png, jpg, jpeg 대응)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 파일 순서 무작위 셔플
        random.seed(42) # 재현성을 위해 시드 고정
        random.shuffle(images)

        # 3. 분할 지점 계산
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        def move_files(file_list, target_set):
            for img_name in file_list:
                # 이미지 경로와 라벨 경로 설정
                img_src = os.path.join(cls_path, img_name)
                lbl_name = os.path.splitext(img_name)[0] + '.txt'
                lbl_src = os.path.join(cls_path, lbl_name)

                # 라벨 파일이 존재하는지 확인
                if not os.path.exists(lbl_src):
                    print(f"경고: {img_name}에 매칭되는 라벨 파일이 없습니다. 건너뜁니다.")
                    continue

                # 대상 경로로 복사 (shutil.copy2 사용)
                shutil.copy2(img_src, os.path.join(output_base_dir, f'{target_set}/images', img_name))
                shutil.copy2(lbl_src, os.path.join(output_base_dir, f'{target_set}/labels', lbl_name))

        # 4. 파일 배분 실행
        print(f"처리 중인 클래스: {cls} (총 {len(images)}장)")
        move_files(train_images, 'train')
        move_files(val_images, 'val')

    print("\n데이터셋 분할 및 통합 완료!")
    print(f"결과물 경로: {os.path.abspath(output_base_dir)}")

# 실행 (경로는 본인의 환경에 맞게 수정하세요)
source_dir = "class_data"  # 'hooked', 'worker' 폴더가 있는 곳
output_dir = "merged_dataset" # 결과물이 저장될 곳
split_dataset(source_dir, output_dir)