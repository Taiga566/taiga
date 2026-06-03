# -*- coding: utf-8 -*-
"""
qwen3vl-image-description.py
-----------------------------------------------
フォルダ内画像に対して、観察VLM（Qwen3-VL）で
人物の見た目に関する属性を文章（作文）で記述し、CSVに保存する。
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    Qwen3VLForConditionalGeneration,
)

# =========================================================
#  ユーザー設定（大域変数）
# =========================================================
#INPUT_DIR = "ClothingAttributeDataset"
INPUT_DIR = "images"
OUTPUT_CSV = "attribute_tags_freetextqwen3test.csv"
META_TXT = "result.meta.description.txt"
MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
USE_4BIT = True
VERBOSE = True

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def list_images_in_dir(folder: Path) -> List[Path]:
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

def build_prompt() -> str:
    return (
        # 役割
        "あなたはバイオメトリクスの専門家です。\n"

        # 指示
        "与えられた画像について観察し、属性文章を作成してください。\n"
        # 条件
        "それぞれの手がかりについて、雰囲気を表す主観的な形容詞もしくは形容詞句を用いて、名詞を修飾した表現で記述してください。（手がかり✕雰囲気）\n"

        # 定義
        "手がかり✕雰囲気とはソフトバイオメトリクスに基づく手がかりに対して雰囲気を表現する記述を追加したものです。\n"
        "属性文章は2つに分類されます。それは【人物属性】と【背景】です。\n"
        "【人物属性】は画像中の観察可能な人物そのものの特徴のことを言い、【背景】は画像中の人物属性以外の周囲の状況に関する情報のことを言います。\n"
        "【人物属性】には、身体・外見・行動的な手がかりの3つの観点があります。\n"
        "身体的な手がかりには、性別・見た目年齢・顔の特徴・体型・全身があり、外見的な手がかりには、服装・身に着けているアクセサリー（眼鏡・サングラス・帽子・バッグ・装飾品など）があり、行動的な手がかりには、歩き方・ジェスチャー・表情があります。\n"
        "【背景】には、時間帯・地域・屋内外か・周囲の混雑度・周囲の物体があります。\n"

        # 出力フォーマット
        "必ず以下のフォーマットのみを出力してください。\n"
        "【人物属性】\n"
        "身体的な手がかり：（性別・見た目年齢・顔の特徴・体型・全身について記述、その際にそれぞれの項目について、「雰囲気を表す主観的な形容詞（句）+名詞」の形式で記述すること）\n"
        "外見的な手がかり：（服装・身に着けているアクセサリー〔眼鏡・サングラス・帽子・バッグ・装飾品など〕について記述、その際にそれぞれの項目について、「雰囲気を表す主観的な形容詞（句）+名詞」の形式で記述すること）\n"
        "行動的な手がかり：（歩き方・ジェスチャー・表情について記述、その際にそれぞれの項目について、「雰囲気を表す主観的な形容詞（句）+名詞」の形式で記述すること）\n"
        "【背景】\n"
        "（時間帯・屋内外か・周囲の混雑度・周囲の物体について記述、その際にそれぞれの項目について、「雰囲気を表す主観的な形容詞（句）+名詞」の形式で記述すること）\n"

        # 記述ルール
        "各項目はそれぞれ100文字程度の自然な日本語の文章でまとめてください。\n"
        "箇条書きやJSONは使用しないでください。\n"
        "フォーマットだけを出力してください。\n"
        "情報が確認できない項目は記述しないでください。\n"

        # 注意事項
        "一歩ずつ順を追って考えてください。\n"
    )

def init_qwen3vl() -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    if USE_4BIT:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPE,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=bnb,
            dtype=DTYPE,
            attn_implementation="sdpa",
        ).eval()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            dtype=DTYPE,
            attn_implementation="sdpa",
        ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model.generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    return model, processor

@torch.inference_mode()
def describe_single_image(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    img_path: Path,
) -> str:
    img = Image.open(img_path).convert("RGB")
    prompt = build_prompt()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    out = model.generate(**inputs, use_cache=True, return_dict_in_generate=True)
    gen_ids = out.sequences[:, inputs["input_ids"].shape[1]:]
    resp = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    del img, inputs, out
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return resp

def write_csv(csv_path: Path, rows: List[Dict[str, str]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["filename", "description"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_meta(meta_path: Path, model: Qwen3VLForConditionalGeneration, prompt: str):
    meta_path = meta_path.resolve()
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = [
        "=== Image Description Meta ===",
        f"MODEL_ID: {MODEL_ID}",
        "OUTPUT_FORMAT: CSV (filename, description)",
        "",
        "=== Prompt ===",
        prompt,
    ]
    meta_path.write_text("\n".join(meta), encoding="utf-8")

def main():
    input_dir = Path(INPUT_DIR).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"入力フォルダが見つかりません: {input_dir}")

    images = list_images_in_dir(input_dir)
    if not images:
        raise FileNotFoundError(f"画像ファイルが見つかりません: {input_dir}")

    out_csv = Path(OUTPUT_CSV).expanduser().resolve()
    meta_txt = Path(META_TXT).expanduser().resolve()

    model, processor = init_qwen3vl()

    prompt = build_prompt()
    write_meta(meta_txt, model, prompt)
    if VERBOSE:
        print(f"[META] saved: {meta_txt}")

    rows: List[Dict[str, str]] = []
    failed: List[str] = []

    for p in images:
        try:
            description = describe_single_image(model, processor, p)
            clean_desc = description.replace("\n", " ").replace("\r", "")
            rows.append({"filename": p.name, "description": clean_desc})
            if VERBOSE:
                print(f"[OK] {p.name} -> {clean_desc[:50]}...")
        except Exception as e:
            rows.append({"filename": p.name, "description": "エラーにより取得失敗"})
            failed.append(f"{p.name}: {e}")
            print(f"[FAIL] {p.name}: {e}")

    write_csv(out_csv, rows)
    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved META: {meta_txt}")
    print(f"Processed: {len(rows)} images (failed: {len(failed)})")

if __name__ == "__main__":
    main()