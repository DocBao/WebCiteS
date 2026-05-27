import json


def run_greedy_min_cover(ias_plan: dict, config: dict = None) -> list:
    """
    【100% 完全召回版】
    不再做任何 Minimum Cover 计算，直接提取 MACU 中涉及到的所有 Doc 编号！
    """
    if not ias_plan or "MACUs" not in ias_plan:
        print("[MinCover] 警告：传入的 ias_plan 为空或没有 MACUs")
        return []

    macus = ias_plan.get("MACUs", [])
    ecus = []

    for idx, macu in enumerate(macus, start=1):
        # 优先读取 supporting_span_ids 列表
        span_ids = macu.get("supporting_span_ids", [])

        # 兼容性兜底：如果没找到，尝试从 supporting_evidence 提取
        if not span_ids and "supporting_evidence" in macu:
            span_ids = [span["span_id"] for span in macu.get("supporting_evidence", [])]

        if not span_ids:
            continue

        # 核心逻辑：提取所有不重复的 source_id
        # 例如 ["Doc_1:S_2", "Doc_2:S_4"] -> ["Doc_1", "Doc_2"]
        all_sources = sorted(list(set(span_id.split(":")[0] for span_id in span_ids)))

        # 记录到数据中
        macu["S_req"] = all_sources

        # 1 MACU = 1 句话，直接挂上所有的交叉引用
        ecus.append({
            "ecu_id": f"ECU_{idx}",
            "MACUs": [macu]
            # "citation_set": all_sources
        })

    return ecus


if __name__ == "__main__":
    print("MinCover 已更新为：100% 完全提取所有涉及的文档。")