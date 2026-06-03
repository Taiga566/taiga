# -*- coding: utf-8 -*-
"""
make_catchcopy_from_attribute_tags.py  (gpt-oss robust v4)
---------------------------------------------------------
attribute_tags.csv を読み込み、標語LLMで3メニュー分のキャッチコピーを生成してCSVに保存。
テンプレ補完なし。失敗は error に記録。

改善点:
- gpt-oss が出す "1) salt ramen..." 等の英語を拾わない
- "(2)" のようなカウント注釈を含む行を除外
- 1)/Line1 の正規抽出は使わず、「候補行フィルタ→3行選択」に統一
"""

# =========================================================
#INPUT_CSV = "attribute_tags.csv"
#INPUT_CSV = "attribute_tags_freetext.csv"
#INPUT_CSV = "attribute_tags_ClothingAttributeDataset_QwenQwen3-VL-32B-Instruct.csv"


#OUTPUT_CSV = "catchcopy.csv"
#MENUS = ["塩ラーメン", "きつねうどん", "かつ丼"]

#SLOGAN_MODEL_ID = "openai/gpt-oss-120b"
#SLOGAN_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
#SLOGAN_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
#SLOGAN_MODEL_ID = "cyberagent/Llama-3.1-70B-Japanese-Instruct-2407"
#SLOGAN_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct" # need to log in
#SLOGAN_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

#SLOGAN_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"


# -*- coding: utf-8 -*-
"""
make_catchcopy_from_description.py
---------------------------------------------------------
作文データ(description列)を読み込み、標語LLMで3メニュー分のキャッチコピーを生成してCSVに保存。
"""

import csv
import re
import sys
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# ユーザー設定
# =========================================================
INPUT_CSV = "attribute_tags_freetextqwen3all1.csv" # 作文が入った新しいCSV
OUTPUT_CSV = "catchcopyllamatoqwen3all1.csv"
MENUS = ["塩ラーメン", "きつねうどん", "かつ丼"]

SLOGAN_MODEL_ID = "cyberagent/Llama-3.1-70B-Japanese-Instruct-2407"
#SLOGAN_MODEL_ID = "Qwen/Qwen3.5-27B"

USE_4BIT_FOR_QWEN = True
MAX_NEW_TOKENS = 160
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.05
FORCE_DETERMINISTIC_FOR_GPTOSS = True
# MAX_RETRIES_PER_IMAGE = 5
VERBOSE = True
# =========================================================

# ---- helpers ----
def _sanitize_no_menu_names(text: str, menus: List[str]) -> str:
    sanitized = text
    for m in menus:
        pattern = rf"[\s\(\[\{{<『「【]*{re.escape(m)}[\s\)\]\}}>』」】:：、，。．]*"
        sanitized = re.sub(pattern, "", sanitized)
    return sanitized.strip()

def _len_ok(s: str) -> bool:
    return 8 <= len(s) <= 12

def _is_gpt_oss(model_id: str) -> bool:
    return model_id.startswith("openai/gpt-oss")

def _build_user_prompt(description: str, retry: int, gpt_oss: bool) -> str:
    base = (
        # 役割
        "あなたはコピーライターです。\n"

        # 指示
        "与えられた【人物属性】と【背景】をもとに、下記の各メニューを薦める状況を想定します。\n"
        "その人物に合ったキャッチコピーを、各メニューについてそれぞれ出力してください。\n"
        "キャッチコピーは【人物属性】の内容を主に反映し、発想の80%以上を人物属性に基づいてください。\n"
        "【背景】は残り20%以内の補助情報として考慮してください。\n"
        "機微情報（年齢/性別/人種/国籍/宗教/身体的特徴/障害/性的指向など）に触れないでください。\n" 

        # 出力ルール
        "各キャッチコピーの文字数は8から12文字。絶対に12文字を超えないこと。日本語のみ。\n"
        "メニュー名（塩ラーメン/きつねうどん/かつ丼）を文中に含めない。\n"
        "キャッチコピーに句読点を含めない。\n"
        "必ず3行のみ出力。説明・理由・前置きは禁止。\n"
        "英字・数字・記号の多用・文字数カウント注釈（例: (2)）は禁止。\n"

        #注意事項
        "一歩ずつ順を追って考えてください。\n"
    )
    if gpt_oss:
        base += "【重要】analysisや英語での説明は禁止。キャッチコピー3行のみ。\n"
    if retry > 0:
        base += "【再生成】3行のみ。各行8〜12文字。日本語のみ。前回より切り口を分け、語尾や表現を変えること。\n"

    menu_lines = "\n".join([f"- {m}" for m in MENUS])
    return (
        f"{base}"
        f"【人物の特徴】\n{description}\n"
        f"【メニュー】\n{menu_lines}\n"
        f"出力:\n"
    )

def _load_slogan_model():
    model_id = SLOGAN_MODEL_ID
    tok_kwargs = {"use_fast": True, "trust_remote_code": True}
    mdl_kwargs = {"trust_remote_code": True, "device_map": "auto"}

    if _is_gpt_oss(model_id):
        from transformers import Mxfp4Config
        mx = Mxfp4Config(dtype="float16")
        mdl_kwargs["quantization_config"] = mx
    else:
        if USE_4BIT_FOR_QWEN:
            from transformers import BitsAndBytesConfig
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            mdl_kwargs["quantization_config"] = bnb
            mdl_kwargs["dtype"] = dtype

    tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, **mdl_kwargs)

    try:
        if getattr(mdl.generation_config, "pad_token_id", None) is None:
            mdl.generation_config.pad_token_id = tok.eos_token_id
    except Exception:
        pass
    return tok, mdl

@torch.inference_mode()
def _generate_only_new_tokens(tok, mdl, system_text: str, user_text: str, do_sample: bool) -> str:
    try:
        messages = [{"role":"system","content":system_text},
                    {"role":"user","content":user_text}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = system_text + "\n" + user_text

    inputs = tok(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        eos_token_id=getattr(tok, "eos_token_id", None),
        repetition_penalty=REPETITION_PENALTY,
    )
    if do_sample:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P

    out = mdl.generate(**inputs, **gen_kwargs)
    gen_ids = out[:, input_len:]
    return tok.decode(gen_ids[0], skip_special_tokens=True).strip()

# --- core filter/select ---
ASCII_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"[0-9０-９]")
COUNT_NOTE_RE = re.compile(r"\(\s*\d+\s*\)")

def _candidate_lines(gen_text: str) -> List[str]:
    lines = [ln.strip() for ln in gen_text.splitlines() if ln.strip()]
    cand = []
    for ln in lines:
        if COUNT_NOTE_RE.search(ln):
            continue
        if "analysis" in ln.lower():
            continue
        if ASCII_RE.search(ln):
            continue
        if DIGIT_RE.search(ln):
            continue

        ln = re.sub(r"^[\-\*\u30fb]+", "", ln).strip()
        ln = re.sub(r"^\d+\)?\s*", "", ln).strip()

        if "：" in ln:
            head, rest = ln.split("：", 1)
            if len(head) <= 8:
                ln = rest.strip()
        elif ":" in ln:
            head, rest = ln.split(":", 1)
            if len(head) <= 8:
                ln = rest.strip()

        ln = _sanitize_no_menu_names(ln, MENUS)
        if not ln:
            continue
        if len(ln) > 30:
            continue
        cand.append(ln)

    seen = set()
    out = []
    for x in cand:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _select_three(cand: List[str]) -> Tuple[str, str, str]:
    good = [c for c in cand if _len_ok(c)]
    rest = [c for c in cand if c not in good]

    picked = []
    for x in good + rest:
        picked.append(x)
        if len(picked) == 3:
            break
    while len(picked) < 3:
        picked.append("")
    return picked[0], picked[1], picked[2]

def _validate(a: str, b: str, c: str) -> Tuple[bool, str]:
    if not (a and b and c):
        return False, "missing_lines"
    if not (_len_ok(a) and _len_ok(b) and _len_ok(c)):
        return False, "length_not_8_12"
    if ASCII_RE.search(a + b + c) or DIGIT_RE.search(a + b + c):
        return False, "contains_ascii_or_digit"
    return True, ""

def _run_one(tok, mdl, description: str, retry: int, gpt_oss: bool) -> Tuple[Tuple[str, str, str], str]:
    system_text = "あなたはコピーライターです。出力は3行のみ。説明禁止。"
    user_text = _build_user_prompt(description, retry, gpt_oss)

    do_sample = DO_SAMPLE
    if gpt_oss and FORCE_DETERMINISTIC_FOR_GPTOSS:
        do_sample = False

    gen = _generate_only_new_tokens(tok, mdl, system_text, user_text, do_sample)
    cand = _candidate_lines(gen)
    a, b, c = _select_three(cand)
    return (a, b, c), gen

def main():
    df = pd.read_csv(INPUT_CSV)
    if "filename" not in df.columns or "description" not in df.columns:
        raise ValueError("CSVに filename または description 列がありません")

    tok, mdl = _load_slogan_model()
    is_gptoss = _is_gpt_oss(SLOGAN_MODEL_ID)

    out_rows = []
    for i, r in df.iterrows():
        filename = str(r["filename"])
        description = str(r.get("description", "")).strip()
        if not description or description == "nan":
            description = "特徴情報なし"

        best = ("", "", "")
        err = ""
        last_raw = ""

        # for retry in range(MAX_RETRIES_PER_IMAGE + 1):
        #     (a, b, c), raw = _run_one(tok, mdl, description, retry, is_gptoss)
        #     last_raw = raw
        #     ok, err_reason = _validate(a, b, c)
        #     best = (a, b, c)
        #     if ok:
        #         err = ""
        #         break
        #     err = err_reason

        (a, b, c), raw = _run_one(tok, mdl, description, 0, is_gptoss)
        last_raw = raw
        ok, err_reason = _validate(a, b, c)
        best = (a, b, c)
        err = "" if ok else err_reason

        if VERBOSE:
            desc_short = description[:30] + "..." if len(description) > 30 else description
            print(f"[{i+1}/{len(df)}] {filename} -> {best} (err={err}) (desc={desc_short})")
            if err:
                print("[DEBUG] raw_generation:\n" + last_raw[:900], file=sys.stderr)

        out_rows.append({
            "filename": filename,
            MENUS[0]: best[0],
            MENUS[1]: best[1],
            MENUS[2]: best[2],
            "error": (err + " | " + last_raw[:200].replace("\n", "\\n")) if err else ""
        })

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename"] + MENUS + ["error"])
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"\nSaved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
