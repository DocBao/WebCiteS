import re
import json


def _split_sentences_zh(text: str) -> list:
    """
    (私有函数) 按中文常见结束标点进行安全分句
    """
    segments = re.split(r'(?<=[。；！？;!?\n])', text)
    sentences = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) > 1:
            sentences.append(seg)
        elif len(sentences) > 0 and len(seg) > 0:
            sentences[-1] += seg
    return sentences


def run_serialize(docs: list, config: dict = None) -> list:
    """
    【纯函数模块入口：数据序列化】
    输入：
        - docs: 原始字符串列表 ["文档1内容...", "文档2内容..."]
        - config: 配置字典 (在此模块可选，为保持接口统一保留)
    输出：
        - list: 序列化后的文档数据结构，包含 source_id 和 spans
    """
    if not docs:
        return []

    serialized_docs = []
    for doc_idx, doc_text in enumerate(docs, start=1):
        source_id = f"Doc_{doc_idx}"
        sentences = _split_sentences_zh(doc_text)

        spans = []
        for span_idx, sentence_text in enumerate(sentences, start=1):
            spans.append({
                "span_id": f"{source_id}:S_{span_idx}",
                "text": sentence_text
            })

        serialized_docs.append({
            "source_id": source_id,
            "spans": spans
        })

    return serialized_docs


# ==========================================
# 单测 (模块自我验证)
# ==========================================
if __name__ == "__main__":
    test_docs = ["第一句话。第二句话！", "另一篇文档的第一句；第二句。"]
    print("运行 Serialize 模块单测...")
    result = run_serialize(test_docs)
    print(json.dumps(result, ensure_ascii=False, indent=2))