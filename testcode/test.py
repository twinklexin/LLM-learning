'''
python testcode/test.py
'''


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread

model_name = "Qwen/Qwen3-4B"

# 设置tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    pad_token_id=tokenizer.pad_token_id,
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# 设置模型为评估模式
model.eval()

# 输入文本
input_text = "请你简单介绍下同济大学机械与能源工程学院"

# 编码输入文本
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    add_special_tokens=True,
)
inputs = {key: value.to(device) for key, value in inputs.items()}

# 生成文本（使用流式输出）
with torch.no_grad():
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=200,
        num_beams=1,              # 改为1以支持流式输出
        temperature=0.7,          
        top_p=0.9,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # 在后台线程中生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("模型生成的文本：")
    for text in streamer:
        print(text, end="", flush=True)
    
    thread.join()
print("\n生成完成！")
