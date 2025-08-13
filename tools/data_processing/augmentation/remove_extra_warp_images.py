import json
import os

def remove_extra_warp_images():
    """JSONL에 없는 42개 이미지를 삭제합니다."""
    warp_path = "/home/intern/banner_vis/data/warp_4500"
    warp_images_path = os.path.join(warp_path, "images")
    warp_jsonl_path = os.path.join(warp_path, "warp_4500.jsonl")
    
    # JSONL에서 이미지 이름들 읽기
    jsonl_images = set()
    with open(warp_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            image_name = data.get('image', '')
            if image_name:
                jsonl_images.add(image_name)
    
    print(f"JSONL에 있는 이미지 수: {len(jsonl_images)}")
    
    # 실제 이미지 파일들 확인
    actual_images = set()
    for file in os.listdir(warp_images_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            actual_images.add(file)
    
    print(f"실제 이미지 파일 수: {len(actual_images)}")
    
    # JSONL에 없는 이미지들 찾기
    missing_in_jsonl = actual_images - jsonl_images
    print(f"JSONL에 없는 이미지 수: {len(missing_in_jsonl)}")
    
    if missing_in_jsonl:
        print("삭제할 이미지들:")
        for img in list(missing_in_jsonl)[:10]:  # 처음 10개만 출력
            print(f"  - {img}")
        
        # JSONL에 없는 이미지들 삭제
        deleted_count = 0
        for image_name in missing_in_jsonl:
            image_path = os.path.join(warp_images_path, image_name)
            try:
                os.remove(image_path)
                deleted_count += 1
                print(f"삭제됨: {image_name}")
            except Exception as e:
                print(f"삭제 실패 {image_name}: {e}")
        
        print(f"\n✅ {deleted_count}개 이미지를 삭제했습니다!")
        
        # 최종 확인
        final_image_count = len([f for f in os.listdir(warp_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"최종 이미지 수: {final_image_count}")
        print(f"JSONL 엔트리 수: {len(jsonl_images)}")
        
        if final_image_count == len(jsonl_images):
            print("🎉 완벽하게 일치합니다!")
        else:
            print("❌ 여전히 불일치가 있습니다.")
    else:
        print("모든 이미지가 JSONL에 있습니다!")

if __name__ == "__main__":
    remove_extra_warp_images() 