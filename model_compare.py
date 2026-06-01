#!/usr/bin/env python3
"""
YOLO 다중 모델 results.csv 비교 스크립트
사용법: python3 compare_results.py --runs_dir ./runs
"""

import argparse
import os
import pandas as pd
from pathlib import Path

# 비교할 주요 지표 (results.csv 컬럼명 기준)
TARGET_METRICS = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "train/box_loss",
    "val/box_loss",
]

def find_results_csv(runs_dir: str) -> dict:
    """각 모델 폴더에서 results.csv 탐색"""
    runs_path = Path(runs_dir)
    found = {}

    for folder in sorted(runs_path.iterdir()):
        if not folder.is_dir():
            continue
        csv_path = folder / "results.csv"
        if csv_path.exists():
            found[folder.name] = csv_path
        else:
            # weights 하위에 있는 경우도 탐색
            for sub_csv in folder.rglob("results.csv"):
                found[folder.name] = sub_csv
                break

    return found


def extract_best_metrics(csv_path: Path, metrics: list) -> dict:
    """results.csv에서 각 지표의 최고값(마지막 epoch 기준) 추출"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # 공백 제거

    result = {}
    for metric in metrics:
        if metric in df.columns:
            if "loss" in metric:
                result[metric] = df[metric].min()   # loss는 최솟값
            else:
                result[metric] = df[metric].max()   # mAP 등은 최댓값
        else:
            result[metric] = None  # 해당 컬럼 없으면 None

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_dir', type=str, default='/home/rokey/runs/detect/runs', help='runs 폴더 경로')
    parser.add_argument('--output',   type=str, default='comparison.csv', help='결과 저장 파일명')
    parser.add_argument('--sort_by',  type=str, default='metrics/mAP50(B)', help='정렬 기준 지표')
    args = parser.parse_args()

    # CSV 탐색
    csv_files = find_results_csv(args.runs_dir)
    if not csv_files:
        print(f"❌ {args.runs_dir} 에서 results.csv를 찾을 수 없습니다.")
        return

    print(f"✅ {len(csv_files)}개 모델 발견: {list(csv_files.keys())}")

    # 지표 추출
    rows = []
    for model_name, csv_path in csv_files.items():
        try:
            metrics = extract_best_metrics(csv_path, TARGET_METRICS)
            metrics["model"] = model_name
            rows.append(metrics)
        except Exception as e:
            print(f"  ⚠️  {model_name} 읽기 실패: {e}")

    # 데이터프레임 생성
    df = pd.DataFrame(rows)
    cols = ["model"] + [c for c in df.columns if c != "model"]
    df = df[cols]

    # 정렬
    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=False).reset_index(drop=True)

    # 출력
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\n📊 모델 비교 결과:")
    print(df.to_string(index=False))

    # CSV 저장
    out_path = os.path.join(args.runs_dir, args.output)
    df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"\n💾 저장 완료: {out_path}")


if __name__ == '__main__':
    main()