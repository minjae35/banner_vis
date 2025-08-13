import json
import os

def add_missing_warp_jsonl():
    """warp JSONL에 없는 42개 이미지를 merged_data_cleaned.jsonl에서 찾아서 추가합니다."""
    warp_path = "/home/intern/banner_vis/data/warp_4500"
    warp_jsonl_path = os.path.join(warp_path, "warp_4500.jsonl")
    warp_images_path = os.path.join(warp_path, "images")
    source_jsonl_path = "/home/intern/banner_vis/json/crop-kanana-label/merged_data_cleaned.jsonl"
    
    # warp JSONL에서 이미지 이름들 읽기
    warp_jsonl_images = set()
    warp_jsonl_data = []
    
    with open(warp_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            image_name = data.get('image', '')
            if image_name:
                warp_jsonl_images.add(image_name)
            warp_jsonl_data.append(data)
    
    print(f"warp JSONL에 있는 이미지 수: {len(warp_jsonl_images)}")
    
    # warp 실제 이미지 파일들 확인
    warp_actual_images = set()
    for file in os.listdir(warp_images_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            warp_actual_images.add(file)
    
    print(f"warp 실제 이미지 파일 수: {len(warp_actual_images)}")
    
    # JSONL에 없는 이미지들 찾기
    missing_in_jsonl = warp_actual_images - warp_jsonl_images
    print(f"warp JSONL에 없는 이미지 수: {len(missing_in_jsonl)}")
    
    if missing_in_jsonl:
        print("JSONL에 추가할 이미지들:")
        for img in list(missing_in_jsonl)[:10]:  # 처음 10개만 출력
            print(f"  - {img}")
        
        # merged_data_cleaned.jsonl에서 누락된 이미지들의 JSONL 찾기
        source_data = {}
        with open(source_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                image_name = data.get('image', '')
                if image_name:
                    source_data[image_name] = data
        
        print(f"source JSONL에서 찾은 이미지 수: {len(source_data)}")
        
        # 누락된 이미지들을 source에서 찾아서 추가
        added_count = 0
        for image_name in missing_in_jsonl:
            if image_name in source_data:
                warp_jsonl_data.append(source_data[image_name])
                added_count += 1
                print(f"추가됨: {image_name}")
            else:
                print(f"source에 없음: {image_name}")
        
        # 수정된 JSONL 저장
        with open(warp_jsonl_path, 'w', encoding='utf-8') as f:
            for entry in warp_jsonl_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"\n✅ {added_count}개 이미지를 JSONL에 추가했습니다!")
        
        # 최종 확인
        final_count = len(warp_jsonl_data)
        print(f"최종 JSONL 엔트리 수: {final_count}")
        
    else:
        print("모든 이미지가 JSONL에 있습니다!")

if __name__ == "__main__":
    add_missing_warp_jsonl() 