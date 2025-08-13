#!/usr/bin/env python3
"""
Convert our dataset format to evaluation format
"""

import json
import argparse
import os

def convert_for_evaluation(input_path, output_path):
    """Convert our format to evaluation format"""
    data = []
    
    # Load our format data
    with open(input_path, 'r', encoding='utf-8') as f:
        if input_path.endswith('.jsonl'):
            # JSONL format
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            # JSON format
            data = json.load(f)
    
    # Convert to evaluation format
    eval_data = []
    for item in data:
        # Extract image path
        image_path = item['image']
        
        # Extract reference text from conversations
        reference_text = ""
        for conv in item['conversations']:
            if conv['from'] == 'gpt':
                # Get the first GPT response (OCR result)
                reference_text = conv['value']
                break
        
        if reference_text:
            eval_data.append({
                'image_path': image_path,
                'reference_text': reference_text
            })
    
    # Save in evaluation format
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(eval_data)} items from {input_path} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert dataset for evaluation")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    convert_for_evaluation(args.input, args.output)

if __name__ == "__main__":
    main() 