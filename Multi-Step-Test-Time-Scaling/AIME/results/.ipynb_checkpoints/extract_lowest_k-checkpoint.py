"""
python extract_lowest_k.py --input aime2024_best_Qwen2.5-Math-1.5B_20251118_205906.jsonl --k 1
python extract_lowest_k.py --input aime2024_best_Qwen2.5-Math-1.5B_20251118_205906.jsonl --k 2
python extract_lowest_k.py --input aime2024_best_Qwen2.5-Math-1.5B_20251118_205906.jsonl --k 4
"""
import json
import argparse

def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records.append(obj)
    return records

def main(args):
    records = load_records(args.input)

    # 按 score_norm 升序排序
    records.sort(key=lambda x: x["score_norm"])

    # 取前 k 个
    lowest_records = records[:args.k]

    out_path = args.output or args.input.replace(".jsonl", f"_lowest_top{args.k}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in lowest_records:
            obj = {
                "question": r["question"],
                "answer": r["best_answer"]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"已保存最低置信度 top-{args.k} 到 {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="scores.jsonl 文件路径")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args)
