#!/usr/bin/env python3
"""
Convert JSONL files to JSON format for evaluation
"""

import json
import argparse
import os

def convert_jsonl_to_json(jsonl_path, json_path):
    """Convert JSONL file to JSON format"""
    data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(data)} items from {jsonl_path} to {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to JSON for evaluation")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    convert_jsonl_to_json(args.input, args.output)

if __name__ == "__main__":
    main() 