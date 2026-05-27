import json
import os
import re
import time
from tqdm import tqdm
from model_use import client, MODEL_NAME

# 带有重试机制的 API 调用封装
def call_api_with_retry(messages, response_format=None, temperature=0.1, max_retries=3):
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temperature
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"\nAPI 调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)  # 等待2秒后重试
    return None


# ==========================================
# 1. 数据序列化 (Serialization)
# ==========================================
def split_sentences_zh(text: str) -> list:
    segments = re.split(r'(?<=[。；！？;!?\n])', text)
    sentences = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) > 1:
            sentences.append(seg)
        elif len(sentences) > 0 and len(seg) > 0:
            sentences[-1] += seg
    return sentences


def serialize_docs(docs: list) -> list:
    serialized = []
    for doc_idx, doc_text in enumerate(docs, start=1):
        source_id = f"Doc_{doc_idx}"
        sentences = split_sentences_zh(doc_text)
        spans = [{"span_id": f"{source_id}:S_{i}", "text": s} for i, s in enumerate(sentences, start=1)]
        serialized.append({"source_id": source_id, "spans": spans})
    return serialized


# ==========================================
# 2. IAS 规划器 (Planner)
# ==========================================
def run_planner(query: str, serialized_docs: list) -> dict:
    context_str = "".join([
        f"\n【来源：{doc['source_id']}】\n" + "".join([f"[{span['span_id']}] {span['text']}\n" for span in doc['spans']])
        for doc in serialized_docs
    ])

    system_prompt = """你是一个严谨的归因摘要规划器（Attribution Planner）。
请根据用户问题和参考文档，制定一个分点的声明计划。
【约束】：
1. 必须输出 JSON 格式。
2. 只能通过 span_id (如 Doc_1:S_2) 引用原文，严禁捏造。
3. 结构必须包含 "query_facets" (列表) 和 "MACUs" (列表，内含 claim_plan, supporting_span_ids, attribution_requirements)。"""

    user_prompt = f"【问题】：\n{query}\n\n【参考文档】：\n{context_str}\n\n请输出严格的 JSON 规划："

    content = call_api_with_retry(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.1
    )

    if not content: return {}

    try:
        raw_ias = json.loads(content)
        # 方案B：物理填充真实 Quote 防幻觉
        span_dict = {span['span_id']: span['text'] for doc in serialized_docs for span in doc['spans']}
        for macu in raw_ias.get("MACUs", []):
            macu["supporting_evidence"] = [
                {"span_id": sid, "quote": span_dict[sid]} for sid in macu.get("supporting_span_ids", []) if
                sid in span_dict
            ]
        return raw_ias
    except Exception as e:
        print(f"\nJSON 解析失败: {e}")
        return {}


# ==========================================
# 3. 确定性来源绑定 (Greedy MinCover)
# ==========================================
def run_greedy_min_cover(ias_plan: dict) -> list:
    macus = ias_plan.get("MACUs", [])
    ecus = []

    for idx, macu in enumerate(macus, start=1):
        supporting_evidence = macu.get("supporting_evidence", [])
        candidate_sources = list(set(span["span_id"].split(":")[0] for span in supporting_evidence))

        if not candidate_sources: continue

        # Specificity 计算：字数越多，细节越丰富
        source_scores = {
            src: sum(len(span["quote"]) for span in supporting_evidence if span["span_id"].startswith(src))
            for src in candidate_sources
        }

        # 贪心选择 Top-1 作为最小必要来源
        ranked_sources = sorted(candidate_sources, key=lambda x: source_scores[x], reverse=True)
        s_req = [ranked_sources[0]]

        macu["candidate_sources"] = candidate_sources
        macu["S_req"] = s_req

        ecus.append({
            "ecu_id": f"ECU_{idx}",
            "MACUs": [macu],
            "citation_set": s_req
        })
    return ecus


# ==========================================
# 4. 上下文感知生成 (Context-Aware Verbalizer)
# ==========================================
def run_verbalizer(ecus: list) -> str:
    generated_summary = ""

    for ecu in ecus:
        current_claims = [macu.get("claim_plan") for macu in ecu.get("MACUs", [])]

        # 提取引用数字，例如 Doc_5 -> [5]
        citation_numbers = sorted([int(doc.replace("Doc_", "")) for doc in ecu.get("citation_set", [])])
        citation_markers = "".join([f"[{num}]" for num in citation_numbers])

        system_prompt = """你是一个“零增量文字转换器（Zero-Delta Verbalizer）”。
【核心约束】
1. 事实忠实：只能表达给定的事实要点，绝对不引入新实体、时间、数值。
2. 上下文连贯：请参考“前文内容”，使用合适的代词或连接词（如“此外”、“其次”），使新句子与前文无缝衔接。
3. 纯净输出：仅输出一句话，且必须在句末带上我要求你加的【引用标记】。不要有任何废话。"""

        claims_text = "\n".join([f"- {c}" for c in current_claims])
        user_prompt = f"【前文内容】:\n{generated_summary if generated_summary else '（尚无前文）'}\n\n"
        user_prompt += f"【本句必须表达的事实要点】:\n{claims_text}\n\n【本句强制引用标记】: {citation_markers}\n\n请输出："

        new_sentence = call_api_with_retry(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.3
        )

        if new_sentence:
            new_sentence = new_sentence.strip()
            # 强制兜底：如果模型没加引用或者加错了，用代码强行修正
            new_sentence = re.sub(r'\[\d+\]', '', new_sentence).strip()
            new_sentence += citation_markers
            generated_summary += new_sentence + " "

    return generated_summary.strip()


# ==========================================
# 主流程 (Main Orchestrator)
# ==========================================
def main():
    input_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test.json"
    output_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_hsbec_final.json"

    if not os.path.exists(input_file):
        print(f"找不到输入文件: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)[:10]

    final_results = []

    print(f"🚀 开始批量处理 {len(data)} 条数据...")

    # 遍历所有数据，加上 tqdm 进度条
    for i, sample in enumerate(tqdm(data, desc="Processing Pipeline")):
        query = sample.get("query", "")
        docs = sample.get("docs", [])

        # 1. 序列化
        serialized_docs = serialize_docs(docs)

        # 2. Planner 规划
        ias_plan = run_planner(query, serialized_docs)

        # 3. MinCover 绑定
        if ias_plan:
            ecus = run_greedy_min_cover(ias_plan)
        else:
            ecus = []

        # 4. Verbalizer 生成
        if ecus:
            predict_summary = run_verbalizer(ecus)
        else:
            predict_summary = "无法生成有效摘要。"

        # 5. 构建对齐 WebCiteS 评测的数据结构
        # 保证 "idx" 在第一行
        out_dict = {
            "idx": i,  # 从 0 开始计数
            "id": sample.get("id", ""),
            "query": query,
            "docs": docs,
            "prompt": sample.get("prompt", ""),
            "summary": sample.get("summary", ""),  # 原始 Label，评测 Claim 时需要
            "predict": predict_summary,  # 生成的带引用摘要
            # 也可以把中间态存下来，方便写论文时做 Case Study 截图
            "hsbec_debug_info": {
                "ECUs": ecus
            }
        }
        final_results.append(out_dict)

        # 每处理完一条就存一下盘，防止中途断网导致数据丢失
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 批量处理完成！所有结果已保存至: {output_file}")
    print("下一步：你可以直接将此文件喂给 src/aqfs/eval.py 运行评测！")


if __name__ == "__main__":
    main()