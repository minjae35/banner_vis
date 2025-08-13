#!/usr/bin/env python3
import json
import os
from pathlib import Path

def recreate_crop_jsonl():
    """기존 minimal_7000.jsonl을 기준으로 새로운 crop jsonl 생성"""
    input_file = "/home/intern/banner_vis/data/base_data/crop_7000/minimal_7000.jsonl"
    output_file = "/home/intern/banner_vis/data/base_data/crop_7000/crop_7000_resized.jsonl"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            data = json.loads(line.strip())
            # 기존 구조 그대로 유지
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Recreated crop jsonl: {len(lines)} entries")

def recreate_flat_jsonl():
    """기존 flat_3500.jsonl을 기준으로 새로운 flat jsonl 생성"""
    input_file = "/home/intern/banner_vis/data/base_data/flat_3500/flat_3500.jsonl"
    output_file = "/home/intern/banner_vis/data/base_data/flat_3500/flat_3500_new.jsonl"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            data = json.loads(line.strip())
            # 기존 구조 그대로 유지
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Recreated flat jsonl: {len(lines)} entries")

def recreate_warp_jsonl():
    """기존 warp_4500.jsonl을 기준으로 새로운 warp jsonl 생성"""
    input_file = "/home/intern/banner_vis/data/base_data/warp_4500/warp_4500.jsonl"
    output_file = "/home/intern/banner_vis/data/base_data/warp_4500/warp_4500_new.jsonl"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            data = json.loads(line.strip())
            # 기존 구조 그대로 유지
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Recreated warp jsonl: {len(lines)} entries")

def main():
    print("Recreating jsonl files from existing ones...")
    
    recreate_crop_jsonl()
    recreate_flat_jsonl()
    recreate_warp_jsonl()
    
    print("Done!")

if __name__ == "__main__":
    main() 