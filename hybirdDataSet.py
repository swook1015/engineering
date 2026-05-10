import os
import shutil
import random
from pathlib import Path

def build_hybrid_dataset(sliced_dir="./final_dataset_overlap50", original_dir="./merged_dataset", output_dir="./final_dataset_hybrid", original_ratio=0.2):
    # 경로를 미리 Path 객체로 확실하게 변환 (문자열 나눗셈 에러 방지)
    sliced_path = Path(sliced_dir)
    orig_path = Path(original_dir)
    out_path = Path(output_dir)

    for split in ["train", "val"]:
        out_img = out_path / split / "images"
        out_lbl = out_path / split / "labels"
        
        # 출력 폴더 생성
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        
        print(f"[{split.upper()}] 데이터 세팅 중...")

        # 1. 슬라이싱 데이터 전부 복사 (안전고리 학습용)
        sliced_img_dir = sliced_path / split / "images"
        if sliced_img_dir.exists():
            for f in sliced_img_dir.iterdir():
                if f.suffix.lower() in ['.jpg', '.png']:
                    shutil.copy(f, out_img / f.name)
                    lbl = sliced_path / split / "labels" / f.with_suffix(".txt").name
                    if lbl.exists():
                        shutil.copy(lbl, out_lbl / lbl.name)

        # 2. 원본 데이터 20% 샘플링 복사 (작업자 학습용)
        orig_img_dir = orig_path / split / "images"
        if orig_img_dir.exists():
            orig_imgs = [f for f in orig_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']]
            # 최소 1장 이상은 샘플링되도록 보장
            sample_n = max(1, int(len(orig_imgs) * original_ratio))
            
            for f in random.sample(orig_imgs, sample_n):
                # 원본 이미지는 이름 앞에 'orig_'를 붙여서 섞음
                shutil.copy(f, out_img / f"orig_{f.name}")
                lbl = orig_path / split / "labels" / f.with_suffix(".txt").name
                if lbl.exists():
                    shutil.copy(lbl, out_lbl / f"orig_{lbl.name}")

    print("✅ 하이브리드 데이터셋 구성이 완벽하게 끝났습니다!")

# 실행
build_hybrid_dataset()