import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 导入你写好的主程序中的核心处理函数
# 假设你在 main_pipeline.py 中定义了 process_single_sample
from main_multi import process_single_sample
from baseline import run_end_to_end_baseline

# 定义我们在主程序中设置的失败标识语
FAILURE_KEYWORDS = [
    "规划失败。",
    "MinCover 计算失败。",
    "生成失败或格式异常。",
    "Baseline 生成失败。"
]

# 线程锁，用于安全地更新和存盘
file_lock = threading.Lock()


def process_and_return_index(list_index: int, sample: dict, config: dict) -> tuple:
    """
    包装一层处理函数，使其在并发返回时能带上它在列表中的索引，方便原位替换
    """
    # 1. 正常调用函数，只接收一个字典变量，不解包！
    processed_sample = process_single_sample(sample, config)

    # 2. 我们通过检查生成的 predict 字段，来自己判断是否成功
    predict_text = processed_sample.get("predict", "")

    # 定义失败关键词
    FAILURE_KEYWORDS = [
        "规划失败。",
        "MinCover 计算失败。",
        "生成失败或格式异常。",
        "Baseline 生成失败。"
    ]

    # 如果预测为空，或者在失败关键词里，说明失败了
    is_success = bool(predict_text) and (predict_text not in FAILURE_KEYWORDS)

    return list_index, processed_sample, is_success


def main():
    # 1. 保持与主程序完全一致的配置
    config = {
        # "api_key": "sk-xxxxxxxxxxxxxxxx",  # 请替换为真实 Key
        # "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # "model_name": "qwen-max",
        "temperature_planner": 0.1,
        "temperature_verbalizer": 0.3,
        "max_retries": 3,
        "max_workers": 1  # 重试时的并发数
    }

    # 你的输出结果文件（也就是我们要扫描和修复的文件）
    output_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_hsbec_final_mC.json"

    if not os.path.exists(output_file):
        print(f"❌ 找不到结果文件: {output_file}。请先运行主程序。")
        return

    # 2. 加载已经处理过的数据
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔍 正在扫描文件，总数据量: {len(data)} 条...")

    # 3. 筛选出所有失败的数据
    failed_items = []  # 存放元组: (在data列表中的索引, 数据本身)

    for i, sample in enumerate(data):
        predict_text = sample.get("predict", "")
        # 如果预测结果为空，或者属于我们定义的失败关键字，则判定为失败
        if not predict_text or predict_text in FAILURE_KEYWORDS:
            failed_items.append((i, sample))

    if not failed_items:
        print("🎉 太棒了！扫描完毕，所有数据均已成功生成，无需重试！")
        return

    # 4. 输出失败的序号列表 (idx)
    failed_idx_list = [sample["idx"] for _, sample in failed_items]
    print(f"\n⚠️ 发现 {len(failed_items)} 条生成失败的数据。")
    print(f"失败的数据序号 (idx) 列表: {failed_idx_list}")

    user_input = input("\n是否立即开始并发重试修复这些数据？(y/n): ")
    if user_input.lower() != 'y':
        print("已取消重试。")
        return

    # 5. 并发重试修复
    print(f"\n🚀 开始并发重试，开启 {config['max_workers']} 个线程...")
    success_count = 0

    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
        # 提交重试任务
        future_to_item = {
            executor.submit(process_and_return_index, list_index, sample, config): list_index
            for list_index, sample in failed_items
        }

        # 实时监控重试进度
        for future in tqdm(as_completed(future_to_item), total=len(failed_items), desc="Retrying"):
            try:
                list_index, processed_sample, is_success = future.result()

                if is_success:
                    success_count += 1

                # 加锁：安全地更新原数据列表并存盘
                with file_lock:
                    data[list_index] = processed_sample
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"\n线程执行发生未捕获异常: {e}")

    # 6. 打印重试总结
    print(f"\n✅ 重试任务结束！")
    print(f"本次成功修复: {success_count}/{len(failed_items)} 条数据。")
    if success_count < len(failed_items):
        print("💡 仍有部分数据失败。你可以再次运行本脚本继续尝试，或者手动检查这些数据的特殊性。")
    print(f"最新的完整结果已更新至: {output_file}")


if __name__ == "__main__":
    main()