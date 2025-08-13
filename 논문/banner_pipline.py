#!/usr/bin/env python3
import os
import argparse
import warnings
import glob
import json
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TypedDict, Tuple

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from flashtext import KeywordProcessor

# =========================
# 환경 변수: GPU 설정 (반드시 torch import 전)
# =========================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

# Huggingface tokenizer warning 억제
warnings.filterwarnings(
    "ignore",
    message="The tokenizer class you load from this checkpoint is not the same type as the class this function is called from.*",
)

# =========================
# 타입 정의
# =========================
class QwenSample(TypedDict):
    id: str
    image: str
    conversations: List[Dict[str, str]]

# =========================
# 설정 데이터 클래스
# =========================
@dataclass
class Config:
    model_name: str = "kakaocorp/kanana-1.5-v-3b-instruct"
    batch_dir: str = "/home/intern/ocr/cropped-banner/image/"
    save_json: str = "/home/intern/banner_vis/data/KonaanData/label/crop.json"
    save_jsonl: Optional[str] = "/home/intern/banner_vis/data/KonaanData/label/crop.jsonl"
    prompt_ocr: str = "현수막에서 보이는 텍스트를 추출해줘."
    guideline_text: str = (
        "현수막 유형 분류 기준은 다음과 같다.\n"
        "1. 정당 현수막: 정당명, 후보자, 선거 관련 문구 포함\n"
        "2. 민간 현수막: 기업/학원/상점/개인/종교/노동조합/동창회/대학 학과/협회 등 민간 단체/개인의 홍보, 행사 안내\n"
        "3. 공공 현수막: 정부·지자체·공공기관 명칭 또는 정책·단속·행정 안내 등 공익적 내용 포함\n"
        "   (단, 단순 '후원'만 공공기관인 경우는 민간으로 본다.)"
    )
    log_file: str = "banner_pipeline.log"
    gen_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(max_new_tokens=512, top_p=1.0, num_beams=1, do_sample=False))
    retry_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(do_sample=True, top_p=0.9, num_beams=1, max_new_tokens=512))
    key_map: Dict[str, List[str]] = field(default_factory=lambda: {
        "정당": ["정당", "후보", "선거", "투표", "공약", "기호"],
        "공공": ["공공", "정부", "지자체", "공단", "공사", "구청", "시청", "동사무소", "주민자치", "정책", "단속", "기념행사", "센터", "문화원", "재활원", "위원회", "재단"],
        "민간": ["기업", "회사", "주식회사", "학원", "아카데미", "세미나", "총회", "야유회", "체육대회", "단합대회", "경시대회", "노동조합",
                 "교회", "성당", "절", "조계종", "부흥회", "선교", "전화", "www", "이벤트", "판매", "무료점검", "세탁", "회원모집", "전시회", "사진전"]
    })
    sponsor_pattern: re.Pattern = re.compile(r"(후원|협찬)\s*:\s*(시청|구청|보건복지부|.*재활원|.*공단|정보통신부|교육인적자원부)")

# =========================
# 파이프라인 클래스
# =========================
class BannerPipeline:
    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model, self.processor = self.load_model_and_processor(config.model_name)
        self.keyword_processor = self.build_keyword_processor(config.key_map)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.log_file, encoding='utf-8')
            ]
        )

    def load_model_and_processor(self, model_name: str) -> Tuple[Any, Any]:
        self.logger.info(f"Loading model and processor: {model_name}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return model, processor

    def build_keyword_processor(self, key_map: Dict[str, List[str]]) -> KeywordProcessor:
        kp = KeywordProcessor(case_sensitive=False)
        for label, keywords in key_map.items():
            for kw in keywords:
                kp.add_keyword(kw, label)
        return kp

    def clean_text(self, text: str) -> str:
        patterns = [
            r"^현수막에서\s*추출된\s*텍스트.*?:\s*",
            r"^다음은\s*현수막에서\s*추출한\s*텍스트.*?:\s*",
        ]
        for pat in patterns:
            text = re.sub(pat, "", text, flags=re.I | re.M)
        lines = []
        for line in text.splitlines():
            line = re.sub(r"^\s*[-*•]+\s*", "", line)
            lines.append(line.replace("**", "").replace("__", ""))
        return "\n".join(lines).strip()

    def build_inputs(self, images: List[Image.Image], prompt: str) -> Dict[str, Any]:
        img_token = getattr(self.processor, "image_token", "<image>")
        samples = [{"image":[img], "conv":[{"role":"user","content":f"{img_token}\n{prompt}"}]} for img in images]
        return self.processor.batch_encode_collate(samples, padding_side="left", add_generation_prompt=True, max_length=8192)

    def build_text_inputs(self, prompt: str) -> Dict[str, Any]:
        samples = [{"image":[], "conv":[{"role":"user","content":prompt}]}]
        return self.processor.batch_encode_collate(samples, padding_side="left", add_generation_prompt=True, max_length=4096)

    def generate_text(self, inputs: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> str:
        inputs = {k:v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
        with torch.inference_mode():
            outs = self.model.generate(**inputs, **gen_kwargs)
        decoded = self.processor.tokenizer.decode(outs[0], skip_special_tokens=True).strip()
        prompt_len = inputs["input_ids"].shape[1]
        if outs.shape[1] > prompt_len:
            decoded = self.processor.tokenizer.decode(outs[0][prompt_len:], skip_special_tokens=True).strip()
        return decoded

    def parse_label(self, raw: str) -> str:
        m = re.search(r"(정당|공공|민간)", raw)
        return (m.group(1) if m else "").strip() or "공공"

    def rule_based_label(self, ocr_text: str) -> str:
        found = set(self.keyword_processor.extract_keywords(ocr_text))
        if "정당" in found: return "정당"
        if "공공" in found: return "공공"
        return "민간"  

    def post_adjust_label(self, ocr_text: str, label: str) -> str:
        if label == "공공" and self.config.sponsor_pattern.search(ocr_text):
            return "민간"
        return label

    def decide_final_label(self, vlm_label: str, rule_label: str, ocr: str) -> str:
        if vlm_label == rule_label:
            return vlm_label
        if "정당" in {vlm_label, rule_label}:
            return "정당"
        sponsor_only = bool(self.config.sponsor_pattern.search(ocr))
        contains_public = any(kw == "공공" for kw in self.keyword_processor.extract_keywords(ocr))
        if vlm_label == "공공" and rule_label == "민간":
            return "민간" if sponsor_only else "공공"
        if vlm_label == "민간" and rule_label == "공공":
            return "공공" if (contains_public and not sponsor_only) else "민간"
        return rule_label

    def run_single(self, img_path: str) -> QwenSample:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        ocr = self.clean_text(
            self.generate_text(
                self.build_inputs([img], self.config.prompt_ocr), self.config.gen_kwargs
            )
        )
        if not ocr:
            ocr = self.clean_text(
                self.generate_text(
                    self.build_inputs([img], self.config.prompt_ocr), self.config.retry_kwargs
                )
            )

        reason_prompt = (
            f"다음은 현수막 OCR 결과다.\n---\n{ocr}\n---\n\n{self.config.guideline_text}\n\n"
            "이 텍스트가 어떤 유형(정당/공공/민간)으로 보이는지 판단한 **근거만** 자세히 써줘. 최종 라벨은 쓰지 마."
        )
        reason = self.clean_text(self.generate_text(self.build_text_inputs(reason_prompt), self.config.gen_kwargs))

        label_prompt = (
            f"다음은 현수막 OCR 텍스트다.\n---\n{ocr}\n---\n\n{self.config.guideline_text}\n\n"
            "위 기준을 적용해 이 현수막이 속하는 최종 라벨을 **정당 / 공공 / 민간** 중 하나로만 답해라."
        )
        vlm_label = self.parse_label(self.clean_text(self.generate_text(self.build_text_inputs(label_prompt), self.config.gen_kwargs)))
        rule_label = self.post_adjust_label(ocr, self.rule_based_label(ocr))
        final_label = self.decide_final_label(vlm_label, rule_label, ocr)

        sid = os.path.splitext(os.path.basename(img_path))[0]
        return {
            "id": sid,
            "image": os.path.basename(img_path),
            "conversations": [
                {"from": "human", "value": f"<image>\n{self.config.prompt_ocr}"},
                {"from": "gpt",   "value": ocr},
                {"from": "human", "value": "이 텍스트를 바탕으로 정당 / 공공 / 민간 중 어떤 유형인지 판단한 근거를 설명해줘."},
                {"from": "gpt",   "value": reason},
                {"from": "human", "value": "최종적으로 이 현수막은 정당 / 공공 / 민간 중 어디에 해당하나요?"},
                {"from": "gpt",   "value": final_label},
            ]
        }

    def run_batch(self) -> List[QwenSample]:
        # 이미지 파일 검색
        files = []
        for ext in ("*.jpg","*.jpeg","*.png","*.webp"):
            files.extend(glob.glob(os.path.join(self.config.batch_dir, ext)))
        files = sorted(files)

        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(self.config.save_json), exist_ok=True)
        if self.config.save_jsonl:
            os.makedirs(os.path.dirname(self.config.save_jsonl), exist_ok=True)
            jsonl_fp = open(self.config.save_jsonl, 'w', encoding='utf-8')
        else:
            jsonl_fp = None

        results: List[QwenSample] = []
        for idx, fp in enumerate(files, 1):
            self.logger.info(f"Processing [{idx}/{len(files)}]: {fp}")
            try:
                rec = self.run_single(fp)
                results.append(rec)
                if jsonl_fp:
                    jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jsonl_fp.flush()
            except Exception:
                self.logger.exception(f"Failed at {fp}")

        if jsonl_fp:
            jsonl_fp.close()

        # 전체 JSON 저장
        with open(self.config.save_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved all results to {self.config.save_json}")
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banner OCR & classification pipeline")
    parser.add_argument("--batch_dir",  type=str, help="Directory with images to process")
    parser.add_argument("--save_json",  type=str, help="Output JSON file path")
    parser.add_argument("--save_jsonl", type=str, help="Output JSONL file path", default=None)
    parser.add_argument("--log_file",   type=str, help="Log file path",      default=Config().log_file)
    args = parser.parse_args()

    cfg = Config(
        batch_dir  = args.batch_dir,
        save_json  = args.save_json,
        save_jsonl = args.save_jsonl,
        log_file   = args.log_file
    )
    pipeline = BannerPipeline(cfg)
    pipeline.run_batch()
