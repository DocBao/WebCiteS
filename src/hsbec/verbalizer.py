import json
import re
import time
from model_use import client, MODEL_NAME


def _build_verbalizer_prompt(query: str, prefix: str, current_claims: list) -> tuple:
    # 动态判断是否为第一句，实现总起句排版
    is_first = (prefix == "")

    system_prompt = """你是一个严谨的“纯文本事实融合器（Zero-Delta Verbalizer）”。
你的任务是将提供的事实要点，完美地融合为自然、流畅的中文表述。

【生死攸关的约束】
1. 事实忠实：只能表达提供的事实要点，绝对不允许补充外部知识！
2. 纯净输出：直接输出纯文本，**绝对不要在句末或句中添加任何 [1]、[2] 这样的引用编号！**（系统会在底层自动添加）。"""

    if is_first:
        system_prompt += """
3. 话语结构（开头总述）：这是回答的第一句话。请结合【用户问题】，在句首自然地给出一小句“总述”（如：“关于此问题，原因有以下几点：”），然后紧接着用“首先，...”或“第一，...”融合本句的事实要点。
4. 强制单句（防断句）：为了配合系统评测，总述与事实之间【必须使用冒号（：）或逗号（，）连接】，你输出的整段话【只能在最末尾包含一个句号（。）】，绝对不能被句号断成两句话！"""
    else:
        system_prompt += """
3. 上下文连贯：请仔细参考“前文内容”，使用合适的连接词（如“此外”、“其次”、“另外”、“最后”），使新生成的句子与前文无缝衔接。
4. 强制单句：你输出的文本【只能包含一个句号（。）】。你可以用逗号或分号连接多个并列的观点，但绝对不能断成多句话。"""

    claims_text = "\n".join([f"- {claim}" for claim in current_claims])

    user_prompt = f"【用户问题】:\n{query}\n\n"
    user_prompt += f"【前文内容】:\n{prefix if prefix else '（尚无前文，这是摘要首句）'}\n\n"
    user_prompt += f"【本句必须融合表达的事实要点】:\n{claims_text}\n\n"
    user_prompt += "请严格按照要求输出纯文本（切记不要自己加引用编号）："

    return system_prompt, user_prompt


def run_verbalizer(query: str, ecus: list, config: dict) -> str:
    if not ecus:
        print("\n[Verbalizer] 警告：接收到的 ecus 为空！")
        return ""

    # client = OpenAI(api_key=config.get("api_key"), base_url=config.get("base_url"))
    model_name = MODEL_NAME
    temperature = config.get("temperature_verbalizer", 0.3)
    max_retries = config.get("max_retries", 3)

    generated_summary = ""

    for ecu in ecus:
        macus = ecu.get("MACUs", [])
        current_claims = [macu.get("claim_plan") for macu in macus]

        # ==========================================
        # 【终极防漏兜底】：从所有可能的地方榨取引用文档
        # ==========================================
        all_sources = set()
        for macu in macus:
            # 1. 尝试从常规字段拿
            all_sources.update(macu.get("candidate_sources", []))
            all_sources.update(macu.get("S_req", []))

            # 2. 绝对兜底：直接从 supporting_span_ids 截取 (确保万无一失！)
            # 只要 planner 写了 "Doc_1:S_2"，这里就必定能提取出 "Doc_1"
            for span_id in macu.get("supporting_span_ids", []):
                all_sources.add(span_id.split(":")[0])

        # # 3. 从 ECU 级别也捞一下
        # for doc in ecu.get("citation_set", []):
        #     all_sources.add(doc.split(":")[0])

        # 清洗、转为数字并排序：过滤掉空值，提取 [1, 2, 5]
        citation_numbers = sorted(list(set([
            int(doc.replace("Doc_", ""))
            for doc in all_sources
            if isinstance(doc, str) and doc.startswith("Doc_")
        ])))

        citation_markers = "".join([f"[{num}]" for num in citation_numbers])

        # 打印 Debug 信息，你在终端可以直接看到是否提取成功！
        # print(f"\n[Verbalizer Debug] 本句提炼出的引用为: {citation_markers}")

        # ==========================================

        sys_p, usr_p = _build_verbalizer_prompt(query, generated_summary, current_claims)

        new_sentence = ""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                    temperature=temperature,
                )
                new_sentence = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"\n[Verbalizer] API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)

        if new_sentence:
            # 清除可能带有的首尾引号（防止结尾是 。" 导致判定失败）
            new_sentence = new_sentence.strip('"').strip("'").strip()

            # 暴力清洗模型叛逆私自生成的残余引用
            new_sentence = re.sub(r'\[\d+(?:[,\-，、]\d+)*\]|【\d+(?:[,\-，、]\d+)*】', '', new_sentence).strip()

            # 【精确强制注入交叉引用】：确保加在中文句号之前
            if citation_markers:
                if new_sentence.endswith("。") or new_sentence.endswith("；"):
                    new_sentence = new_sentence[:-1] + citation_markers + new_sentence[-1]
                else:
                    new_sentence += citation_markers + "。"
            else:
                if not new_sentence.endswith("。"):
                    new_sentence += "。"

            generated_summary += new_sentence + " "

    return generated_summary.strip()


if __name__ == "__main__":
    print("Verbalizer 已更新为：终极防漏兜底提取引用 + 总述句支持。")