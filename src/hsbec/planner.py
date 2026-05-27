import json
import time
from openai import OpenAI
from model_use import client, MODEL_NAME


# ==========================================
# 内部辅助函数 1：构造 Prompt
# ==========================================
def _build_planner_prompt(query: str, serialized_docs: list) -> tuple:
    """
    (私有函数) 将序列化后的文档构建为带标识的上下文
    """
    context_str = ""
    for doc in serialized_docs:
        context_str += f"\n【来源：{doc['source_id']}】\n"
        for span in doc['spans']:
            context_str += f"[{span['span_id']}] {span['text']}\n"

    system_prompt = """你是一个严谨的归因摘要规划器（Attribution Planner）。
    你的任务是根据用户的问题和提供的参考文档，制定一个分点的声明计划（MACUs）。

    【极度重要：穷尽式证据召回】
    针对每一个观点，你必须**穷尽式地**在所有参考文档中寻找证据！只要其他文档中也提到了相似的观点，你必须将它们的所有 span_id 全部加入 `supporting_span_ids` 列表中！**决不能只摘录一篇文档就结束。**

    【严格约束】
    1. 你必须以 JSON 格式输出。
    2. 你不能自己捏造证据，只能通过 `span_id`（如 Doc_1:S_2）来引用原文。
    3.你提取的 MACU 必须是直接回答用户问题的最核心事实。请剔除所有背景介绍、寒暄、举例说明和边缘信息。如果一个观点不痛不痒，请不要将它纳入 MACU！
    4. 你的输出必须严格符合以下 JSON 结构：
{
  "query_facets": ["问题方面1", "问题方面2"],
  "MACUs": [
    {
      "claim_plan": "这里用一句话简述你要生成的观点",
      "supporting_span_ids": ["Doc_1:S_2", "Doc_3:S_1"], 
      "attribution_requirements": ["entity: 实体名称", "value: 具体数值/原因"]
    }
  ]
}"""

    user_prompt = f"【问题】：\n{query}\n\n【参考文档】：\n{context_str}\n\n请输出严格的 JSON 规划："
    return system_prompt, user_prompt


# ==========================================
# 内部辅助函数 2：填充真实 Quote 防幻觉
# ==========================================
def _populate_quotes_from_spans(ias_plan: dict, serialized_docs: list) -> dict:
    """
    (私有函数) 大模型只输出 span_id，我们用代码把真实原文（Quote）填进去。
    """
    span_dict = {}
    for doc in serialized_docs:
        for span in doc['spans']:
            span_dict[span['span_id']] = span['text']

    for macu in ias_plan.get("MACUs", []):
        supporting_evidence = []
        for span_id in macu.get("supporting_span_ids", []):
            if span_id in span_dict:
                supporting_evidence.append({
                    "span_id": span_id,
                    "quote": span_dict[span_id]
                })
        macu["supporting_evidence"] = supporting_evidence

    return ias_plan


# ==========================================
# 核心对外接口：执行 Planner
# ==========================================
def run_planner(query: str, serialized_docs: list, config: dict) -> dict:
    """
    【纯函数模块入口】
    输入：
        - query: 用户查询问题
        - serialized_docs: 序列化后的文档列表
        - config: 包含 API 配置的字典
    输出：
        - dict: 完整的 IAS 结构（填充了真实 quote），若失败返回空字典 {}
    """
    # 1. 初始化客户端 (利用传入的配置)
    # client = OpenAI(
    #     api_key=config.get("api_key"),
    #     base_url=config.get("base_url")
    # )
    # model_name = config.get("model_name", "qwen-max")
    temperature = config.get("temperature_planner", 0.1)
    max_retries = config.get("max_retries", 3)

    # 2. 构造 Prompt
    sys_p, usr_p = _build_planner_prompt(query, serialized_docs)
    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": usr_p}
    ]

    # 3. 带有重试机制的 API 调用
    raw_ias = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            content = response.choices[0].message.content
            raw_ias = json.loads(content)
            break  # 成功则跳出循环
        except Exception as e:
            print(f"[Planner 模块] API调用或JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)

    # 4. 验证并填充真实 Quote
    if raw_ias:
        completed_ias = _populate_quotes_from_spans(raw_ias, serialized_docs)
        return completed_ias
    else:
        return {}


# ==========================================
# 仅用于模块自我测试 (单独运行此脚本时才会执行)
# ==========================================
if __name__ == "__main__":
    # 这里写死测试数据，仅供调试当前模块使用
    test_config = {
        # "api_key": "sk-xxxxxxxxxxxx",  # 填入你的真实 key 用于单测
        # "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # "model_name": "qwen-max",
        "temperature_planner": 0.1,
        "max_retries": 1
    }
    test_query = "文职为什么离职率高"
    test_docs = [
        {"source_id": "Doc_1", "spans": [{"span_id": "Doc_1:S_1", "text": "工作压力大是离职主因。"}]}
    ]

    print("正在运行 Planner 模块单测...")
    result = run_planner(test_query, test_docs, test_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))