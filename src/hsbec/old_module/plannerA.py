import json
import os
import re
from openai import OpenAI
from model_use import client, MODEL_NAME


# ==========================================
# 2. 核心逻辑：构造 Prompt 与调用 API
# ==========================================
def build_planner_prompt(query: str, serialized_docs: list) -> str:
    """
    将序列化后的文档构建为带标识的上下文，输入给大模型
    """
    context_str = ""
    for doc in serialized_docs:
        context_str += f"\n【来源：{doc['source_id']}】\n"
        for span in doc['spans']:
            context_str += f"[{span['span_id']}] {span['text']}\n"

    system_prompt = """你是一个严谨的归因摘要规划器（Attribution Planner）。
你的任务是根据用户的问题和提供的参考文档，制定一个分点的声明计划（MACUs）。

【严格约束】
1. 你必须以 JSON 格式输出。
2. 你不能自己捏造证据，只能通过 `span_id`（如 Doc_1:S_2）来引用原文。
3. 你的输出必须严格符合以下 JSON 结构：
{
  "query_facets": ["问题方面1", "问题方面2"],
  "MACUs": [
    {
      "claim_plan": "这里用一句话简述你要生成的观点",
      "supporting_span_ids": ["Doc_1:S_2", "Doc_3:S_1"], 
      "attribution_requirements": ["entity: 实体名称", "value: 具体数值/原因"]
    }
  ]
}
注意：supporting_span_ids 列表里只能填写文档中真实存在的 span_id！"""

    user_prompt = f"【问题】：\n{query}\n\n【参考文档】：\n{context_str}\n\n请输出严格的 JSON 规划："

    return system_prompt, user_prompt


def generate_ias_from_llm(system_prompt: str, user_prompt: str) -> dict:
    """调用 Qwen API 获取 JSON 格式的规划"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},  # 强制输出 JSON
            temperature=0.1,  # 保持低温度，降低随机性
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"API 调用或 JSON 解析失败: {e}")
        return None


# ==========================================
# 3. 方案 B 核心：利用 span_id 填充真实 Quote
# ==========================================
def populate_quotes_from_spans(ias_plan: dict, serialized_docs: list) -> dict:
    """
    这是方案 B 的灵魂：
    大模型只输出 span_id，我们用代码把真实原文（Quote）填进去。彻底杜绝幻觉！
    """
    # 建立查找字典: span_id -> text
    span_dict = {}
    for doc in serialized_docs:
        for span in doc['spans']:
            span_dict[span['span_id']] = span['text']

    # 遍历 LLM 生成的 MACUs，填充证据
    for macu in ias_plan.get("MACUs", []):
        supporting_evidence = []
        for span_id in macu.get("supporting_span_ids", []):
            if span_id in span_dict:
                supporting_evidence.append({
                    "span_id": span_id,
                    "quote": span_dict[span_id]  # 物理级别的 100% 忠实原文
                })
            else:
                print(f"警告：模型幻觉了一个不存在的 span_id: {span_id}")

        # 将我们填充的真实证据写入计划中
        macu["supporting_evidence"] = supporting_evidence

    return ias_plan


# ==========================================
# 主函数
# ==========================================
def main():
    input_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_serialized.json"
    output_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_ias_plan.json"

    if not os.path.exists(input_file):
        print("请先运行 serialize.py 生成序列化数据！")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 为了跑通流程，我们先拿【第一条数据】进行测试
    sample = data[0]
    query = sample["query"]
    serialized_docs = sample["serialized_docs"]

    print(f"正在为问题：【{query}】生成 IAS 规划...")

    # 1. 构造 Prompt
    sys_p, usr_p = build_planner_prompt(query, serialized_docs)

    # 2. 请求 Qwen 获得初步的 JSON 规划
    raw_ias = generate_ias_from_llm(sys_p, usr_p)

    if raw_ias:
        # 3. 方案 B：利用 span_id 填充原始 quote，完成真正的 IAS
        completed_ias = populate_quotes_from_spans(raw_ias, serialized_docs)

        # 把生成的 IAS 保存回 sample 里
        sample["IAS"] = completed_ias

        # 保存结果用于检查
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([sample], f, ensure_ascii=False, indent=4)

        print("\n【生成的中间归因态 (IAS) 样例】:")
        print(json.dumps(completed_ias, ensure_ascii=False, indent=2))
        print(f"\n✅ 成功！IAS 数据已保存至 {output_file}")


if __name__ == "__main__":
    main()