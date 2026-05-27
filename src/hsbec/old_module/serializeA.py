import json
import re
import os


def split_sentences_zh(text: str) -> list:
    """
    针对中文文本的分句函数（借鉴 WebCiteS 的 eval_utils.py 逻辑）
    """
    # 按照中文常见结束标点进行分割
    segments = re.split(r'(?<=[。；！？;!?\n])', text)
    sentences = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) > 1:  # 正常的句子
            sentences.append(seg)
        elif len(sentences) > 0 and len(seg) > 0:
            # 如果切分出来只有标点符号，拼接到上一句
            sentences[-1] += seg
    return sentences


def serialize_sample(sample: dict) -> dict:
    """
    将单个数据样例的 docs 进行切分与序列化
    """
    docs = sample.get("docs", [])
    serialized_docs = []

    # 遍历每篇文档，doc_idx 从 1 开始，与原始数据中的 [1], [2] 对应
    for doc_idx, doc_text in enumerate(docs, start=1):
        source_id = f"Doc_{doc_idx}"
        sentences = split_sentences_zh(doc_text)

        spans = []
        for span_idx, sentence_text in enumerate(sentences, start=1):
            span_id = f"{source_id}:S_{span_idx}"
            spans.append({
                "span_id": span_id,
                "text": sentence_text
            })

        serialized_docs.append({
            "source_id": source_id,
            "spans": spans
        })

    # 将序列化后的结果添加回原来的字典中
    sample["serialized_docs"] = serialized_docs
    return sample


def main():
    # 假设你的数据路径（请根据实际情况修改）
    input_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test.json"
    output_file = r"C:\Users\XKBei\Desktop\Tools\project\WebCiteS\data\data\aqfs_snippet\test_serialized.json"

    if not os.path.exists(input_file):
        print(f"找不到输入文件: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"正在处理 {len(data)} 条数据...")

    serialized_data = []
    for sample in data:
        serialized_sample = serialize_sample(sample)
        serialized_data.append(serialized_sample)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, ensure_ascii=False, indent=4)

    print(f"序列化完成！结果已保存至: {output_file}")

    # 打印第一条数据的第一篇文档，检查序列化结果
    print("\n【序列化结果样例】:")
    sample_doc = serialized_data[0]["serialized_docs"][0]
    print(json.dumps(sample_doc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()