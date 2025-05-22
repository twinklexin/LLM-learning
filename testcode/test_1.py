from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# prepare the model input
prompt = "简单介绍一下同济大学机械与能源工程学院"
messages = [{"role": "user", "content": prompt}]

# 应用聊天模板，将消息转换为模型可以理解的格式
text = tokenizer.apply_chat_template(
    messages,                    # 输入的消息列表，包含角色和内容
    tokenize=False,             # 不进行分词，返回字符串而不是token
    add_generation_prompt=True,  # 添加生成提示符，使模型知道需要生成回复
    enable_thinking=False,        # 启用思维模式，让模型先思考再回答
)

# 将处理后的文本转换为模型输入张量
model_inputs = tokenizer(
    [text],                     # 将文本转换为列表形式
    return_tensors="pt"         # 返回PyTorch张量格式
).to(model.device)              # 将输入数据移动到模型所在的设备(GPU/CPU)


# 使用模型生成文本
generated_ids = model.generate(
    **model_inputs,              # 展开输入参数
    max_new_tokens=32768,        # 设置最大生成的token数量
    temperature=0.7,             # 控制生成的随机性（小写）
    top_p=0.8,                  # 控制采样的概率阈值（小写）
    top_k=20,                   # 控制每一步保留的最高概率token数量（小写）
    min_p=0,                    # 最小生成概率（小写）
)
# 获取生成的token序列，去除输入部分，只保留新生成的内容
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()


# 解析思维内容（思考过程）
try:
    # 查找</think>标记对应的token ID (151668)
    # [::-1]表示反转列表，从后往前查找第一个</think>标记
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    # 如果没有找到</think>标记，将index设为0
    index = 0

# 解码思维内容：从开始到</think>标记之前的内容
thinking_content = tokenizer.decode(
    output_ids[:index],          # 切片获取思维部分的token
    skip_special_tokens=True     # 跳过特殊token（如PAD、EOS等）
).strip("\n")                    # 去除首尾换行符

# 解码最终回答内容：从</think>标记之后到结束的内容
content = tokenizer.decode(
    output_ids[index:],          # 切片获取回答部分的token
    skip_special_tokens=True     # 跳过特殊token
).strip("\n")                    # 去除首尾换行符

print("thinking content:", thinking_content)
print("content:", content)
