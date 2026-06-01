#!/usr/bin/env python3
"""
YOLO 다중 버전 & 사이즈 학습 스크립트
지원: YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, YOLO26
사용법: python3 train_all_yolo.py --data data.yaml --epochs 100 --batch 16
"""

import argparse
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────
# 학습할 모델 목록 (버전 x 사이즈)
# 필요없는 줄은 주석처리 하세요
# ─────────────────────────────────────────
MODELS = [
    # # YOLOv8
    # "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    # # YOLOv9
    # "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",
    # # YOLOv10
    # "yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x",
    # # YOLOv11
    # "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    # # YOLOv12
    # "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",
    # # YOLO26 (ultralytics 최신)
    # "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",

    # "yolov8n",
    # # YOLOv9
    # "yolov9t",
    # # YOLOv10
    # "yolov10n",
    # # YOLOv11
    # "yolo11n",
    # # YOLOv12
    # "yolo12n", 
    # # YOLO26 (ultralytics 최신)
    # "yolo26n"
    "yolov10n"
]


def get_save_dir(base_dir: str, model_name: str, use_date: bool) -> str:
    if use_date:
        date_str = datetime.now().strftime("%Y%m%d")
        return os.path.join(base_dir, f"{date_str}_{model_name}")
    else:
        return os.path.join(base_dir, model_name)


def train_model(model_name: str, args) -> dict:
    """단일 모델 학습 실행, 결과 반환"""
    from ultralytics import YOLO

    save_dir = get_save_dir(args.save_dir, model_name, args.use_date)
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  🚀 학습 시작: {model_name}")
    print(f"  저장경로: {save_dir}")
    print(f"{'='*60}")

    try:
        model = YOLO(f"{model_name}.pt")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            project=args.save_dir,
            name=os.path.basename(save_dir),
            exist_ok=False,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr,
            seed=args.seed,
            verbose=False,
        )
        elapsed = time.time() - start_time
        print(f"  ✅ 완료: {model_name} ({elapsed/60:.1f}분)")
        return {"model": model_name, "status": "success", "time": elapsed}

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ 실패: {model_name} → {e}")
        return {"model": model_name, "status": "failed", "error": str(e), "time": elapsed}


def print_summary(results: list, total_time: float):
    print(f"\n{'='*60}")
    print(f"  📊 학습 결과 요약")
    print(f"{'='*60}")
    success = [r for r in results if r["status"] == "success"]
    failed  = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print(f"  ✅ 성공 : {len(success)}개")
    for r in success:
        print(f"     - {r['model']} ({r['time']/60:.1f}분)")

    if skipped:
        print(f"  ⏭️  스킵  : {len(skipped)}개")
        for r in skipped:
            print(f"     - {r['model']} (이미 완료)")

    if failed:
        print(f"  ❌ 실패 : {len(failed)}개")
        for r in failed:
            print(f"     - {r['model']}: {r.get('error', '')}")

    print(f"\n  ⏱️  총 소요시간: {total_time/60:.1f}분")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='YOLO 다중 버전 학습 스크립트')
    parser.add_argument('--data',      type=str,   required=True,        help='데이터셋 yaml 경로 (필수)')
    parser.add_argument('--epochs',    type=int,   default=500,          help='에폭 수 (기본: 100)')
    parser.add_argument('--batch',     type=int,   default=16,           help='배치 사이즈 (기본: 16)')
    parser.add_argument('--imgsz',     type=int,   default=640,          help='이미지 사이즈 (기본: 640)')
    parser.add_argument('--device',    type=str,   default='0',          help='GPU 장치 (기본: 0, CPU: cpu)')
    parser.add_argument('--workers',   type=int,   default=8,            help='데이터로더 워커 수 (기본: 8)')
    parser.add_argument('--patience',  type=int,   default=100,           help='Early stopping patience (기본: 100)')
    parser.add_argument('--optimizer', type=str,   default='auto',       help='옵티마이저 (기본: auto)')
    parser.add_argument('--lr',        type=float, default=0.01,         help='초기 학습률 (기본: 0.01)')
    parser.add_argument('--save_dir',  type=str,   default='./runs',     help='결과 저장 폴더 (기본: ./runs)')
    parser.add_argument('--use_date',  action='store_true',              help='폴더명에 날짜 포함 (예: 20260601_yolov8n)')
    parser.add_argument('--skip_done', action='store_true',              help='이미 학습된 모델 스킵')
    parser.add_argument('--models',    type=str,   default=None,         help='학습할 모델 직접 지정 (쉼표구분, 예: yolov8n,yolo11s)')
    parser.add_argument('--dry_run',   action='store_true',              help='실제 학습 없이 목록만 출력')
    parser.add_argument('--seed',      type=int,   default=0,           help='랜덤 시드 (기본: 0)')

    args = parser.parse_args()

    # 모델 목록 결정
    model_list = MODELS
    if args.models:
        model_list = [m.strip() for m in args.models.split(',')]

    print(f"\n{'='*60}")
    print(f"  🎯 YOLO 다중 버전 학습 시작")
    print(f"  데이터셋 : {args.data}")
    print(f"  에폭     : {args.epochs}")
    print(f"  배치     : {args.batch}")
    print(f"  이미지   : {args.imgsz}")
    print(f"  GPU      : {args.device}")
    print(f"  모델수   : {len(model_list)}개")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN] 학습할 모델 목록:")
        for i, m in enumerate(model_list, 1):
            print(f"  {i:2d}. {m}")
        return

    results = []
    total_start = time.time()

    for model_name in model_list:
        # 이미 완료된 모델 스킵
        if args.skip_done:
            save_dir = get_save_dir(args.save_dir, model_name, args.use_date)
            weights_path = os.path.join(save_dir, "weights", "best.pt")
            if os.path.exists(weights_path):
                print(f"  ⏭️  스킵: {model_name} (이미 완료)")
                results.append({"model": model_name, "status": "skipped", "time": 0})
                continue

        result = train_model(model_name, args)
        results.append(result)

    total_time = time.time() - total_start
    print_summary(results, total_time)

    # 결과 로그 저장
    log_path = os.path.join(args.save_dir, "train_summary.txt")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"학습 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 소요시간: {total_time/60:.1f}분\n\n")
        for r in results:
            f.write(f"{r['model']}: {r['status']} ({r['time']/60:.1f}분)\n")
            if r.get("error"):
                f.write(f"  오류: {r['error']}\n")
    print(f"  📄 결과 로그: {log_path}")


if __name__ == '__main__':
    main()