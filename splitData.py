import os
import random
import shutil

def split_dataset_safe(source_base_dir, output_base_dir, split_ratio=0.8):
    dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for d in dirs:
        os.makedirs(os.path.join(output_base_dir, d), exist_ok=True)

    classes = [d for d in os.listdir(source_base_dir) if os.path.isdir(os.path.join(source_base_dir, d))]
    
    for cls in classes:
        cls_path = os.path.join(source_base_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        random.seed(42)
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        def move_files(file_list, target_set, class_prefix):
            for img_name in file_list:
                img_src = os.path.join(cls_path, img_name)
                lbl_name = os.path.splitext(img_name)[0] + '.txt'
                lbl_src = os.path.join(cls_path, lbl_name)

                if not os.path.exists(lbl_src):
                    continue

                # 핵심: 파일 이름 앞에 클래스 이름을 붙여서 중복 방지
                new_name = f"{class_prefix}_{img_name}"
                new_lbl_name = f"{class_prefix}_{lbl_name}"

                shutil.copy2(img_src, os.path.join(output_base_dir, f'{target_set}/images', new_name))
                shutil.copy2(lbl_src, os.path.join(output_base_dir, f'{target_set}/labels', new_lbl_name))

        print(f"처리 중인 클래스: {cls} (총 {len(images)}장)")
        move_files(train_images, 'train', cls) # 클래스 이름 전달
        move_files(val_images, 'val', cls)

    print("\n[성공] 이름 중복 방지 처리 완료!")

source_dir = "class_data" 
output_dir = "merged_dataset"
split_dataset_safe(source_dir, output_dir)