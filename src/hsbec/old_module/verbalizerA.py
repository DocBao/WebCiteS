import json
import os
from model_use import client, MODEL_NAME

# ==========================================
# 2. 构造 Verbalizer Prompt
# ==========================================
def build_verbalizer_prompt(prefix: str, current_claims: list, citation_markers: str) -> tuple:
    """
    构造上下文感知的 Zero-Delta 生成 Prompt
    :param prefix: 之前已经生成的摘要文本
    :param current_claims: 当前 ECU 需要表达的核心事实（MACUs的claim_plan）
    :param citation_markers: 必须强制添加的引用标记，例如 "[5]"
    """
    system_prompt = """你是一个严格的“零增量文字转换器（Zero-Delta Verbalizer）”。
你的任务是将提供的事实要点，转换为一句自然、流畅的中文句子。

【核心约束】
1. 事实忠实（Zero-Delta）：只能表达提供的事实要点，**绝对不允许**引入任何新的实体、时间、数值或因果关系。
2. 上下文连贯（Context-Aware）：请参考“前文内容”，使用合适的代词或连接词（如“此外”、“其次”、“因此”），使新生成的句子与前文无缝衔接。
3. 强制引用：你必须在你生成的这句（或这几句）话的**最末尾**，严格原样附上我提供的【引用标记】。
4. 纯净输出：只需输出转换后的句子即可，不要有任何多余的解释、Markdown 格式或前言后语。"""

    # 格式化当前需要表达的事实
    claims_text = "\n".join([f"- {claim}" for claim in current_claims])

    # 构造用户输入
    user_prompt = "【前文内容】:\n"
    user_prompt += prefix if prefix else "（这是摘要的第一句话，尚无前文）"
    user_prompt += f"\n\n【本句必须表达的事实要点】:\n{claims_text}"
    user_prompt += f"\n\n【本句强制引用标记】: {citation_markers}"
    user_prompt += "\n\n请输出转换后的句子，并在句末带上引用标记："

    return system_prompt, user_prompt


# ==========================================
# 3. 调用大模型进行单句生成
# ==========================================
def generate_sentence(system_prompt: str, user_prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # 稍微给一点温度，让语句更自然连贯
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API 调用失败: {e}")
        return ""


# ==========================================
# 主函数：逐句生成并拼接
# ==========================================
def main():
    input_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_ias_bound.json"
    output_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_hsbec_final.json"

    if not os.path.exists(input_file):
        print(f"找不到输入文件: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data[0]
    ecus = sample.get("IAS", {}).get("sentence_ECUs", [])

    if not ecus:
        print("没有找到 Sentence-ECUs，请检查上一步的输出。")
        return

    print("开始上下文感知的交替生成 (Context-Aware Verbalization)...\n")

    generated_summary = ""

    # 遍历每一个 ECU，逐句生成
    for ecu in ecus:
        ecu_id = ecu.get("ecu_id")
        # 1. 提取当前 ECU 的所有 claim plan
        current_claims = [macu.get("claim_plan") for macu in ecu.get("MACUs", [])]

        # 2. 将 "Doc_5" 格式转换为数字引用 "[5]"
        citation_set = ecu.get("citation_set", [])
        citation_numbers = sorted([int(doc.replace("Doc_", "")) for doc in citation_set])
        citation_markers = "".join([f"[{num}]" for num in citation_numbers])

        print(f"正在生成 {ecu_id} (引用: {citation_markers})...")

        # 3. 构造 Prompt
        sys_p, usr_p = build_verbalizer_prompt(
            prefix=generated_summary,
            current_claims=current_claims,
            citation_markers=citation_markers
        )

        # 4. 调用模型生成当前句子
        new_sentence = generate_sentence(sys_p, usr_p)

        # 5. [简单的 Binding Verification] 确保模型乖乖加上了引用标记
        if not new_sentence.endswith(citation_markers):
            # 如果大模型忘了加，或者加错了，我们用代码强制加上！(绝对的防守)
            # 移除错误结尾的可能引用
            import re
            new_sentence = re.sub(r'\[\d+\]', '', new_sentence).strip()
            new_sentence += citation_markers

        print(f"生成的句子: {new_sentence}\n")

        # 6. 拼接到全局摘要中，作为下一句的 Prefix
        generated_summary += new_sentence + " "

    # 将最终生成的摘要保存回 sample
    sample["predict"] = generated_summary.strip()

    # 为了适配 WebCiteS 的自动评测，我们需要确保数据结构有 "predict" 字段
    # (可以保留原有信息方便对比)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([sample], f, ensure_ascii=False, indent=4)

    print("=" * 50)
    print("【最终生成的 H-SBEC 摘要】:\n")
    print(generated_summary.strip())
    print("=" * 50)
    print(f"\n✅ 成功！最终摘要已保存至 {output_file}")
    print("你可以直接用这个文件去跑 WebCiteS 的 eval.py 评测脚本了！")


if __name__ == "__main__":
    main()