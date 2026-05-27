import json
import os
from tqdm import tqdm
from model_use import client, MODEL_NAME
# 导入你之前写的四个独立模块的核心函数
# （假设你已经在对应的 .py 文件中封装好了这些入口函数）
from serialize import run_serialize
from planner import run_planner
from min_cover_base1 import run_greedy_min_cover
from verbalizer import run_verbalizer


def process_single_sample(sample: dict, config: dict) -> dict:
    """
    处理单条数据的标准 Pipeline，随时可以在这里打断点调试某一个模块
    """
    query = sample.get("query", "")
    docs = sample.get("docs", [])

    # 模块 1：序列化
    serialized_docs = run_serialize(docs)

    # 模块 2：生成 IAS 规划 (如果这步报错，直接返回空)
    ias_plan = run_planner(query, serialized_docs, config)
    if not ias_plan:
        return sample, False

    # 模块 3：计算 MinCover 与构建 Sentence-ECU
    ecus = run_greedy_min_cover(ias_plan)
    if not ecus:
        return sample, False

    # 模块 4：交替生成最终摘要
    predict_summary = run_verbalizer(ecus, config)

    # 构建评测所需的返回格式
    sample["predict"] = predict_summary
    sample["hsbec_debug_info"] = {
        "serialized_docs": serialized_docs,
        "IAS_plan": ias_plan,
        "ECUs": ecus
    }

    return sample, True


def main():
    # 1. 集中管理配置
    config = {
        # "api_key": "sk-xxxxxxxxxxxxxxxx",
        # "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # "model_name": "qwen-max",
        "temperature_planner": 0.1,
        "temperature_verbalizer": 0.3,
        "max_retries": 3
    }

    input_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test.json"
    output_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_hsbec_finalmC.json"

    if not os.path.exists(input_file):
        print(f"找不到输入文件: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_results = []
    success_count = 0

    print(f"🚀 开始模块化处理 Pipeline，共 {len(data)} 条数据...")

    for i, sample in enumerate(tqdm(data, desc="Processing")):
        # 为每条数据增加 idx (与 WebCiteS 评测对齐)
        formatted_sample = {
            "idx": i,
            "id": sample.get("id", ""),
            "query": sample.get("query", ""),
            "docs": sample.get("docs", []),
            "prompt": sample.get("prompt", ""),
            "summary": sample.get("summary", ""),
        }

        # 调用核心 Pipeline
        processed_sample, is_success = process_single_sample(formatted_sample, config)

        if is_success:
            success_count += 1
        else:
            processed_sample["predict"] = "生成失败或格式异常。"

        final_results.append(processed_sample)

        # 实时存盘，防止意外中断
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 运行结束！成功率: {success_count}/{len(data)}")
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    main()