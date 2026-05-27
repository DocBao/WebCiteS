import json
import os


def calculate_specificity(source_id: str, supporting_evidence: list) -> int:
    """
    计算启发式 Specificity 分数（对应论文 6.3 节的 \lambda_spec Specificity(s)）。
    这里的简单实现是：该 source_id 提供的 quote 文本总长度越长，说明细节越丰富。
    """
    score = 0
    for span in supporting_evidence:
        if span["span_id"].startswith(source_id):
            score += len(span["quote"])
    return score


def greedy_min_cover(macu: dict) -> list:
    """
    确定性贪心算法求解最小必要来源集合 (S_req)
    """
    supporting_evidence = macu.get("supporting_evidence", [])

    # 1. 提取所有候选的 source_id (S_cand)
    candidate_sources = list(set(
        span["span_id"].split(":")[0] for span in supporting_evidence
    ))

    if not candidate_sources:
        return []

    # 2. 论文逻辑：如果多个 Source 都能覆盖该 Claim，我们挑选 Specificity 最高的
    # 计算每个 candidate 的分数
    source_scores = {}
    for src in candidate_sources:
        source_scores[src] = calculate_specificity(src, supporting_evidence)

    # 按照分数从高到低排序
    ranked_sources = sorted(candidate_sources, key=lambda x: source_scores[x], reverse=True)

    # 3. 贪心选择：因为当前的 MACU 是原子语义，通常 Top-1 就能完整覆盖
    # 为了避免 over-citation，我们只取排名第 1 的文档作为最小必要来源
    s_req = [ranked_sources[0]]

    return s_req


def build_sentence_ecus(macus: list) -> list:
    """
    将计算好 S_req 的 MACUs 打包为句子级的执行单元 (Sentence-ECUs)
    这里采用最稳妥的策略：1 个 MACU 对应 1 个 Sentence-ECU，
    方便后续 Verbalizer 逐句生成并做 Zero-Delta 校验。
    """
    ecus = []
    for idx, macu in enumerate(macus, start=1):
        s_req = greedy_min_cover(macu)

        # 将结果写入 MACU
        macu["candidate_sources"] = list(set(span["span_id"].split(":")[0] for span in macu["supporting_evidence"]))
        macu["S_req"] = s_req

        # 封装为 Sentence-ECU
        ecu = {
            "ecu_id": f"ECU_{idx}",
            "MACUs": [macu],
            "citation_set": s_req  # ECU 的最终引用集合
        }
        ecus.append(ecu)
    return ecus


def main():
    input_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_ias_plan.json"
    output_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_ias_bound.json"

    if not os.path.exists(input_file):
        print(f"找不到输入文件: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理第一条数据
    sample = data[0]
    ias = sample.get("IAS", {})
    macus = ias.get("MACUs", [])

    print("正在执行 Requirement-level Source Binding (Greedy MinCover)...")

    # 构造 Sentence-ECUs，并计算 S_req
    sentence_ecus = build_sentence_ecus(macus)

    # 更新 IAS
    sample["IAS"]["sentence_ECUs"] = sentence_ecus

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([sample], f, ensure_ascii=False, indent=4)

    print("\n【Source Binding 完成后的 Sentence-ECU 样例】:")
    # 打印第一个 ECU 看看效果
    print(json.dumps(sentence_ecus[0], ensure_ascii=False, indent=2))
    print(f"\n✅ 成功！带有 S_req 的数据已保存至 {output_file}")


if __name__ == "__main__":
    main()