import json
import re
import time
from model_use import client, MODEL_NAME


def _build_verbalizer_prompt(prefix: str, current_claims: list) -> tuple:
    system_prompt = """你是一个严谨的“纯文本事实融合器（Zero-Delta Verbalizer）”。
你的任务是将提供的事实要点，完美地融合为【一句】自然、流畅的中文长句。

【生死攸关的约束】
1. 事实忠实：只能表达提供的事实要点，绝对不允许自我发挥！
2. 上下文连贯（Context-Aware）：请参考“前文内容”，使用合适的代词或连接词（如“此外”、“其次”、“因此”），使新生成的句子与前文无缝衔接,回答需要简洁清晰、逻辑连贯。
3. 强制单句：你输出的文本【只能包含一个句号（。）】。你可以用逗号或分号连接多个并列的观点，但绝对不能断成多句话。
4. 纯净输出：直接输出这句纯文本，**绝对不要在句末或句中添加任何 [1]、[2] 这样的引用编号！**（系统会在底层自动添加）。"""

    claims_text = "\n".join([f"- {claim}" for claim in current_claims])

    user_prompt = f"【前文内容】:\n{prefix if prefix else '（尚无前文，这是摘要首句）'}\n\n"
    user_prompt += f"【本句必须融合表达的事实要点】:\n{claims_text}\n\n"
    user_prompt += "请输出融合后的【单句纯文本】（切记不要自己加引用编号）："

    return system_prompt, user_prompt


def run_verbalizer(ecus: list, config: dict) -> str:
    if not ecus:
        print("\n[Verbalizer] 警告：接收到的 ecus 为空！")
        return ""

    # client = OpenAI(api_key=config.get("api_key"), base_url=config.get("base_url"))
    # model_name = config.get("model_name", "qwen-max")
    temperature = config.get("temperature_verbalizer", 0.3)
    max_retries = config.get("max_retries", 3)

    generated_summary = ""

    for ecu in ecus:
        macus = ecu.get("MACUs", [])
        current_claims = [macu.get("claim_plan") for macu in macus]

        # ==========================================
        # 【核心修改】：不读 citation_set，直接读 candidate_sources 取并集！
        # ==========================================
        all_candidate_sources = set()
        for macu in macus:
            # 拿到该 MACU 所有的 candidate_sources
            sources = macu.get("candidate_sources", [])
            all_candidate_sources.update(sources)

        # 转换为数字并排序：["Doc_1", "Doc_5", "Doc_2"] -> [1, 2, 5]
        citation_numbers = sorted(
            [int(doc.replace("Doc_", "")) for doc in all_candidate_sources if doc.startswith("Doc_")])
        citation_markers = "".join([f"[{num}]" for num in citation_numbers])

        sys_p, usr_p = _build_verbalizer_prompt(generated_summary, current_claims)

        new_sentence = ""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                    temperature=temperature,
                )
                new_sentence = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"\n[Verbalizer] API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)

        if new_sentence:
            # 暴力清洗大模型可能自己乱加的残余引用 (如 [1], 【1,2】)
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
        else:
            print(f"\n[Verbalizer] 警告：ECU {ecu.get('ecu_id')} 生成为空！")

    return generated_summary.strip()


if __name__ == "__main__":
    print("Verbalizer 已更新为：直接读取 candidate_sources 注入全量交叉引用。")