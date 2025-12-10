import json

# 读取原始 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']

# 筛选出 gpt_response 非空的样本并转换格式
def convert_to_jsonl_format(data):
    records = []
    for item in data:
        # gpt_resp_str = item.get('gpt_response')
        # if not gpt_resp_str:  # 如果不存在或为空字符串
        #     continue

        # # 解析 gpt_response JSON
        # try:
        #     gpt_resp = json.loads(gpt_resp_str)
        # except json.JSONDecodeError:
        #     continue  # 跳过无法解析的条目

        # key_steps = gpt_resp.get("key_steps", [])
        # # 检查 key_steps 是否为空或无内容
        # if not key_steps:
        #     continue

        record = {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            # "key_steps": key_steps
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
    input_file = 'TTT_data/Best_of_4_Transformers_Step_Certainty_lambda_0.5_gsm8k_filtered_step_reward_key2.json'
    output_file = 'TTT_data/Bo4_self_certainty.jsonl'

    # 加载数据
    data = load_json(input_file)

    # 筛选并转换格式
    jsonl_data = convert_to_jsonl_format(data)

    # 保存结果
    save_jsonl(jsonl_data, output_file)
    print(f"✅ 已生成 {len(jsonl_data)} 条有效记录，保存到 {output_file}")
