import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from model_use import client, MODEL_NAME

# ==========================================
# 配置：与主程序保持完全一致
# ==========================================
# API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"  # 替换为你的真实 Key
# BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# MODEL_NAME = "qwen-max"
MAX_WORKERS = 10
MAX_RETRIES = 3

# client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
file_lock = threading.Lock()


def run_end_to_end_baseline(list_index: int, sample: dict) -> tuple:
    """
    运行传统的端到端 Baseline：直接用数据集自带的 prompt 请求大模型
    """
    # WebCiteS 数据集里已经帮我们构建好了 prompt (包含了 instruction、带编号的 docs 和 query)
    raw_prompt = sample.get("prompt", "")

    if not raw_prompt:
        return list_index, sample, False

    # 构建极简系统提示词（模拟直接回答）
    system_prompt = ""

    predict_summary = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_prompt}
                ],
                temperature=0.3,  # 常规端到端生成温度
            )
            predict_summary = response.choices[0].message.content.strip()
            break
        except Exception as e:
            time.sleep(2)

    if predict_summary:
        sample["predict"] = predict_summary
        return list_index, sample, True
    else:
        sample["predict"] = "Baseline 生成失败。"
        return list_index, sample, False


def main():
    input_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test.json"
    output_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_base_qwen27.json"

    if not os.path.exists(input_file):
        print(f"找不到输入文件: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🚀 开始运行端到端 Baseline (End-to-End)，共 {len(data)} 条数据...")

    # 格式化数据，保持与评测脚本对齐
    formatted_data = []
    for i, sample in enumerate(data):
        formatted_data.append({
            "idx": i + 1,
            "id": sample.get("id", ""),
            "query": sample.get("query", ""),
            "docs": sample.get("docs", []),
            "prompt": sample.get("prompt", ""),
            "summary": sample.get("summary", ""),
        })

    # 初始化空列表文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f)

    final_results = []
    success_count = 0

    # 并发请求 Baseline
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(run_end_to_end_baseline, i, sample): i
            for i, sample in enumerate(formatted_data)
        }

        for future in tqdm(as_completed(future_to_item), total=len(formatted_data), desc="Baseline Processing"):
            try:
                list_index, processed_sample, is_success = future.result()
                if is_success:
                    success_count += 1

                with file_lock:
                    final_results.append(processed_sample)
                    # 排序存盘
                    final_results.sort(key=lambda x: x["idx"])
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(final_results, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"线程执行异常: {e}")

    print(f"\n🎉 Baseline 运行结束！成功率: {success_count}/{len(data)}")
    print(f"Baseline 结果已保存至: {output_file}")
    print("👉 现在你可以拿这个结果去跑 eval.py，然后和 H-SBEC 的结果做对比了！")


if __name__ == "__main__":
    main()