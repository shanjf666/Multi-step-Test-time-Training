import json

# 读取原始 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 筛选数据
def filter_data(data, threshold):
    filtered_results = [entry for entry in data['results'] if entry['max_confidence'] >= threshold]
    return filtered_results

# 保存为新 JSON 文件
def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# 主程序
if __name__ == "__main__":
    input_file = 'TTT_data/Best_of_4_Transformers_Step_Certainty_lambda_0.5_deepseek_7b_key_2.json'
    output_file = 'TTT_data/filtered_data_threshold_4.9.json'
    threshold = 4.9

    # 加载数据
    data = load_json(input_file)

    # 筛选数据
    filtered_data = filter_data(data, threshold)

    # 保存结果
    save_json(filtered_data, output_file)
    print(f"筛选后的数据已保存到 {output_file}")