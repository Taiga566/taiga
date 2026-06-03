import pandas as pd

FILE_PATH = "catchcopygemma4togemma4all1.csv"
SEARCH_WORD = "男"
EXCLUDE_COLUMNS = {"filename"}


def main():
    df = pd.read_csv(FILE_PATH)

    target_columns = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    if not target_columns:
        raise ValueError("検索対象の列が見つかりません。")

    mask = pd.Series(False, index=df.index)
    hit_columns = []

    for col in target_columns:
        col_mask = df[col].astype(str).str.contains(SEARCH_WORD, na=False)
        if col_mask.any():
            hit_columns.append(col)
        mask = mask | col_mask

    result = df[mask].copy()

    if result.empty:
        print(f"'{SEARCH_WORD}' を含む行は見つかりませんでした。")
        print(f"対象ファイル: {FILE_PATH}")
        print(f"検索対象列: {target_columns}")
        return

    print(f"対象ファイル: {FILE_PATH}")
    print(f"検索ワード: {SEARCH_WORD}")
    print(f"ヒットした列: {hit_columns}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
