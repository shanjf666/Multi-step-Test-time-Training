import json

# 读取原始 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']

# 筛选 + 转换格式
def filter_and_convert(data, threshold):
    records = []
    for item in data:
        # 1. 置信度筛选
        if item.get('max_confidence') > threshold:
            continue

        # 2. 解析 gpt_response
        gpt_resp_str = item.get('gpt_response')
        if not gpt_resp_str:
            continue
        try:
            gpt_resp = json.loads(gpt_resp_str)
        except json.JSONDecodeError:
            continue

        # 3. 提取关键字段
        key_steps = gpt_resp.get("key_steps", [])
        if not key_steps:
            continue

        record = {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "key_steps": key_steps
        }
        records.append(record)
    return records

# 保存为 jsonl 文件
def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

# 主程序
if __name__ == "__main__":
    input_file = 'TTT_data/Best_of_4_Transformers_Step_Certainty_lambda_0.5_gsm8k_filtered_step_reward_key.json'
    threshold = 5.60
    output_file = 'TTT_data/filtered_data_threshold_5.60.jsonl'

    # 加载原始数据
    data = load_json(input_file)

    # 筛选 + 转换
    filtered_records = filter_and_convert(data, threshold)

    # 保存结果
    save_jsonl(filtered_records, output_file)

    print(f"筛选并转换后的数据共 {len(filtered_records)} 条，已保存到 {output_file}")
