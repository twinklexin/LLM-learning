"""
python app_flask.py
"""

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)  # 创建Flask实例

# 加载模型和分词器
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 分词器
model = AutoModelForCausalLM.from_pretrained(model_name)  # 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.route("/generate", methods=["POST"])  # 定义POST路由"/generate"
def generate():
    prompt = request.json.get("prompt")  # 从POST请求中获取prompt参数
    if not prompt:
        return (
            jsonify({"error": "No prompt provided"}),
            400,
        )  # 如果没有提供prompt，返回400错误

    inputs = tokenizer(prompt, return_tensors="pt").to(
        device
    )               # 将输入文本转换为模型可用的张量
    with torch.no_grad():                       # 关闭梯度计算，节省内存
        outputs = model.generate(
            **inputs,
            max_length=200,                     # 生成文本的最大长度
            num_beams=5,                        # 束搜索的束数
            no_repeat_ngram_size=2,             # 防止重复生成相同的n元组
            early_stopping=True                 # 启用早停
        )
    generated_text = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )  # 将生成的token解码为文本
    return jsonify({"generated_text": generated_text})  # 返回生成的文本


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
