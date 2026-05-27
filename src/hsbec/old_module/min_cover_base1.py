import json


def _calculate_specificity(source_id: str, supporting_evidence: list, config: dict) -> float:
    """
    (私有函数) 计算启发式 Specificity 分数
    目前使用最简单的策略：该 Source 提供的 quote 文本总长度越长，细节越丰富。
    """
    # 预留给未来通过 config 传入权重 (如 config.get("lambda_spec", 1.0))
    lambda_spec = config.get("lambda_spec", 1.0) if config else 1.0

    score = 0
    for span in supporting_evidence:
        if span["span_id"].startswith(source_id):
            score += len(span["quote"])
    return score * lambda_spec


def _greedy_min_cover(macu: dict, config: dict) -> list:
    """
    (私有函数) 贪心算法求解最小必要来源集合 S_req
    """
    supporting_evidence = macu.get("supporting_evidence", [])
    candidate_sources = list(set(span["span_id"].split(":")[0] for span in supporting_evidence))

    if not candidate_sources:
        return []

    # 计算每个 candidate 的分数
    source_scores = {
        src: _calculate_specificity(src, supporting_evidence, config)
        for src in candidate_sources
    }

    # 贪心选择 Top-1 覆盖 (原子语义通常一处来源即可证实)
    ranked_sources = sorted(candidate_sources, key=lambda x: source_scores[x], reverse=True)
    return [ranked_sources[0]]


def run_greedy_min_cover(ias_plan: dict, config: dict = None) -> list:
    """
    【纯函数模块入口：确定性来源绑定】
    输入：
        - ias_plan: Planner 模块输出的中间归因态 (包含 MACUs)
        - config: 配置字典 (包含算法权重等)
    输出：
        - list: 构建好的 Sentence-ECUs 列表，包含最小必要来源 S_req
    """
    if not ias_plan or "MACUs" not in ias_plan:
        return []

    macus = ias_plan.get("MACUs", [])
    ecus = []

    for idx, macu in enumerate(macus, start=1):
        s_req = _greedy_min_cover(macu, config)

        # 将算法结果写回 MACU 结构
        macu["candidate_sources"] = list(
            set(span["span_id"].split(":")[0] for span in macu.get("supporting_evidence", [])))
        macu["S_req"] = s_req

        # 封装为 Sentence-ECU
        ecu = {
            "ecu_id": f"ECU_{idx}",
            "MACUs": [macu],
            "citation_set": s_req  # 该句最终要加的引用标签集合
        }
        ecus.append(ecu)

    return ecus


# ==========================================
# 单测 (模块自我验证)
# ==========================================
if __name__ == "__main__":
    test_ias = {
        "MACUs": [{
            "claim_plan": "测试观点",
            "supporting_evidence": [
                {"span_id": "Doc_1:S_1", "quote": "短证据"},
                {"span_id": "Doc_2:S_1", "quote": "这是一条非常长且非常具体的完美证据。"}
            ]
        }]
    }
    print("运行 MinCover 模块单测...")
    result = run_greedy_min_cover(test_ias)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 预期结果：S_req 应该是 ["Doc_2"]