import json
import time
# from openai import OpenAI
from model_use import client, MODEL_NAME


def _build_reflection_prompt(query: str, current_claims: list, serialized_docs: list) -> tuple:
    """
    构造覆盖反思 (Coverage Reflection) 的 Prompt
    """
    # 构造文档上下文
    context_str = ""
    for doc in serialized_docs:
        context_str += f"\n【来源：{doc['source_id']}】\n"
        for span in doc['spans']:
            context_str += f"[{span['span_id']}] {span['text']}\n"

    # 提取当前已有的观点
    claims_text = "\n".join([f"- {claim}" for claim in current_claims]) if current_claims else "（无）"

    system_prompt = """你是一个严谨的“归因覆盖反思器（Coverage Reflection Agent）”。
你的任务是对比【用户问题】与【目前已提取的观点】，检查是否遗漏了回答问题所需的关键事实。

【工作逻辑与严格约束】
1. 对比分析：仔细阅读用户问题，看看【目前已提取的观点】是否已经完整回答了问题。
2. 查漏补缺：如果发现有遗漏的重要方面（Facets），请在【参考文档】中寻找证据并提取为新的观点。
3. 宁缺毋滥：如果目前的观点已经足够回答问题，或者参考文档中没有更多相关信息了，请不要强行凑数！
4. 格式约束：你必须输出 JSON 格式。只能使用文档中真实的 `span_id`。

输出 JSON 结构如下：
{
  "is_missing_facets": true/false,  // 是否发现了遗漏的关键信息
  "missing_analysis": "简要分析遗漏了什么（如果没遗漏填无）",
  "MACUs": [ // 如果 is_missing_facets 为 false，这个列表请留空 []
    {
      "claim_plan": "补充提取的观点",
      "supporting_span_ids": ["Doc_2:S_3"]
    }
  ]
}"""

    user_prompt = f"【用户问题】:\n{query}\n\n【目前已提取的观点】:\n{claims_text}\n\n【参考文档】:\n{context_str}\n\n请进行反思并输出 JSON："

    return system_prompt, user_prompt


def _populate_quotes_from_spans(macus: list, serialized_docs: list) -> list:
    """用真实的原文替换 span_id，防止幻觉"""
    span_dict = {span['span_id']: span['text'] for doc in serialized_docs for span in doc['spans']}

    for macu in macus:
        supporting_evidence = []
        for span_id in macu.get("supporting_span_ids", []):
            if span_id in span_dict:
                supporting_evidence.append({
                    "span_id": span_id,
                    "quote": span_dict[span_id]
                })
        macu["supporting_evidence"] = supporting_evidence
    return macus


def run_coverage_reflection(query: str, serialized_docs: list, ias_plan: dict, config: dict) -> dict:
    """
    【纯函数模块入口：覆盖反思】
    输入：
        - query: 用户查询
        - serialized_docs: 序列化后的文档
        - ias_plan: Planner 刚刚生成的初始计划
        - config: 配置字典
    输出：
        - dict: 补充完缺失 MACUs 的最新 IAS 计划
    """
    if not ias_plan:
        return ias_plan

    current_macus = ias_plan.get("MACUs", [])
    current_claims = [m.get("claim_plan") for m in current_macus]

    # client = OpenAI(api_key=config.get("api_key"), base_url=config.get("base_url"))
    model_name = MODEL_NAME
    temperature = config.get("temperature_planner", 0.1)  # 保持与 Planner 相同的低温度
    max_retries = config.get("max_retries", 3)

    sys_p, usr_p = _build_reflection_prompt(query, current_claims, serialized_docs)

    raw_reflection = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            raw_reflection = json.loads(response.choices[0].message.content)
            break
        except Exception as e:
            print(f"[Reflection] API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)

    if raw_reflection and raw_reflection.get("is_missing_facets"):
        new_macus = raw_reflection.get("MACUs", [])
        if new_macus:
            # 物理填充防幻觉
            new_macus = _populate_quotes_from_spans(new_macus, serialized_docs)
            # 把新发现的观点追加到原有的 IAS 计划中
            ias_plan["MACUs"].extend(new_macus)
            # 记录反思日志，方便 DEBUG
            ias_plan["coverage_reflection_log"] = raw_reflection.get("missing_analysis")

    return ias_plan


if __name__ == "__main__":
    print("Coverage Reflection 模块准备就绪。")