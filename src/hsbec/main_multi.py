import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 导入你写好的模块化函数
from serialize import run_serialize
from planner import run_planner
from min_cover import run_greedy_min_cover
from verbalizer import run_verbalizer
from reflection import run_coverage_reflection # 引入模块

# 创建一个全局线程锁，用于多线程安全地写入 JSON 文件
file_lock = threading.Lock()


def process_single_sample(sample: dict, config: dict) -> dict:
    """
    独立处理单条数据的 Pipeline (无需任何修改，保持之前的纯函数逻辑)
    """
    query = sample.get("query", "")
    docs = sample.get("docs", [])

    # 模块 1：序列化
    serialized_docs = run_serialize(docs)

    # 模块 2：生成 IAS 规划
    ias_plan = run_planner(query, serialized_docs, config)
    if not ias_plan:
        sample["predict"] = "规划失败。"
        return sample
    # ================= 新增：模块 2.5 覆盖反思 =================
    ias_plan = run_coverage_reflection(query, serialized_docs, ias_plan, config)
    # ==========================================================

    # 模块 3：计算 MinCover
    ecus = run_greedy_min_cover(ias_plan, config)
    if not ecus:
        sample["predict"] = "MinCover 计算失败。"
        return sample

    # 模块 4：交替生成最终摘要
    predict_summary = run_verbalizer(query,ecus, config)
    sample["predict"] = predict_summary

    # 调试信息 (可选保存)
    sample["hsbec_debug_info"] = {
        "serialized_docs": serialized_docs,
        "IAS_plan": ias_plan,
        "ECUs": ecus
    }

    return sample


def main():
    config = {
        # "api_key": "sk-xxxxxxxxxxxxxxxx",
        # "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # "model_name": "qwen-max",
        "temperature_planner": 0.1,
        "temperature_verbalizer": 0.3,
        "max_retries": 3,
        # 新增并发配置：根据你的 API 账号等级调整
        "max_workers": 10,
        "recall_threshold": 0,  # [阀门控制] 0.0=引用所有来源(高Recall低Precision)；1.0=只取最高分(低Recall高Precision)
        "max_citations_per_ecu": 3  # [打包控制] 控制合成一句长句的最大引用数量，直接影响 AIS 分数
    }

    input_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test.json"
    output_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_hsbec_final_mCR.json"

    if not os.path.exists(input_file):
        print(f"找不到输入文件: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)[:10]

    # 初始化输出文件为空列表
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f)

    print(f"🚀 开始并发处理，开启 {config['max_workers']} 个线程，共 {len(data)} 条数据...")

    final_results = []
    success_count = 0

    # 格式化输入数据
    formatted_data = []
    for i, sample in enumerate(data):
        formatted_data.append({
            "idx": i,
            "id": sample.get("id", ""),
            "query": sample.get("query", ""),
            "docs": sample.get("docs", []),
            "prompt": sample.get("prompt", ""),
            "summary": sample.get("summary", ""),
        })

    # ==========================================
    # 核心：使用 ThreadPoolExecutor 进行并发请求
    # ==========================================
    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
        # 将任务提交到线程池
        future_to_sample = {
            executor.submit(process_single_sample, sample, config): sample
            for sample in formatted_data
        }

        # 使用 tqdm 包装 as_completed，实时显示并发进度
        for future in tqdm(as_completed(future_to_sample), total=len(formatted_data), desc="Processing"):
            try:
                # 获取该线程的处理结果
                processed_sample = future.result()

                if processed_sample.get("predict") not in ["规划失败。", "MinCover 计算失败。"]:
                    success_count += 1

                # 线程安全地实时存盘 (加锁)
                with file_lock:
                    final_results.append(processed_sample)
                    # 由于 as_completed 是乱序返回的，我们在存盘前可以按 idx 排个序，保证文件整洁
                    final_results.sort(key=lambda x: x["idx"])
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(final_results, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"\n线程执行发生未捕获异常: {e}")

    print(f"\n🎉 并发运行结束！成功率: {success_count}/{len(data)}")
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    main()