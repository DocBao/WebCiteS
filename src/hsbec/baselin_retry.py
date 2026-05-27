import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from model_use import client, MODEL_NAME

# ==========================================
# 1. API 配置 (请与你的 run_baseline.py 保持一致)
# ==========================================

MODEL_NAME = MODEL_NAME
MAX_WORKERS = 4  # 重试时的并发数
MAX_RETRIES = 3


file_lock = threading.Lock()

# 失败的判定关键字
FAILURE_KEYWORDS = ["Baseline 生成失败。", ""]


def process_baseline_retry(list_index: int, sample: dict) -> tuple:
    """
    重试单条 Baseline 数据，返回在原列表中的索引、更新后的样本以及是否成功。
    """
    raw_prompt = sample.get("prompt", "")
    if not raw_prompt:
        return list_index, sample, False

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
                temperature=0.3,
            )
            predict_summary = response.choices[0].message.content.strip()
            break
        except Exception as e:
            # 静默等待并重试
            time.sleep(2)

    # 判定是否成功
    if predict_summary and predict_summary not in FAILURE_KEYWORDS:
        sample["predict"] = predict_summary
        return list_index, sample, True
    else:
        sample["predict"] = "Baseline 生成失败。"
        return list_index, sample, False


def main():
    # 你的 Baseline 结果文件路径
    output_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_base_qwen27.json"

    if not os.path.exists(output_file):
        print(f"❌ 找不到 Baseline 结果文件: {output_file}")
        return

    # 2. 读取当前进度
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔍 正在扫描 Baseline 文件，总数据量: {len(data)} 条...")

    # 3. 筛选出失败的数据
    failed_items = []  # 存储格式: (列表中的索引, 样本数据)
    for i, sample in enumerate(data):
        predict_text = sample.get("predict", "").strip()
        if not predict_text or predict_text in FAILURE_KEYWORDS:
            failed_items.append((i, sample))

    if not failed_items:
        print("🎉 太棒了！扫描完毕，所有 Baseline 数据均已成功生成，无需重试！")
        return

    failed_idx_list = [sample["idx"] for _, sample in failed_items]
    print(f"\n⚠️ 发现 {len(failed_items)} 条生成失败的数据。")
    print(f"失败的数据序号 (idx) 列表: {failed_idx_list}")

    user_input = input("\n是否立即开始并发重试修复这些数据？(y/n): ")
    if user_input.lower() != 'y':
        print("已取消重试。")
        return

    print(f"\n🚀 开始并发重试，开启 {MAX_WORKERS} 个线程...")
    success_count = 0

    # 4. 并发重试并原位替换
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(process_baseline_retry, list_index, sample): list_index
            for list_index, sample in failed_items
        }

        for future in tqdm(as_completed(future_to_item), total=len(failed_items), desc="Retrying Baseline"):
            try:
                list_index, processed_sample, is_success = future.result()

                if is_success:
                    success_count += 1

                # 加锁：安全地覆盖原列表的对应位置并存盘
                with file_lock:
                    data[list_index] = processed_sample
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"\n线程执行发生未捕获异常: {e}")

    # 5. 结算面板
    print(f"\n✅ Baseline 重试任务结束！")
    print(f"本次成功修复: {success_count}/{len(failed_items)} 条数据。")
    if success_count < len(failed_items):
        print("💡 仍有部分数据失败，可以直接再次运行本脚本继续尝试。")
    print(f"完整结果已更新至: {output_file}")


if __name__ == "__main__":
    main()