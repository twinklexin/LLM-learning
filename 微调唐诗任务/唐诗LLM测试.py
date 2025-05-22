import os
import sys
import argparse
import json
import warnings
import logging

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
import transformers
from peft import PeftModel
from colorama import Fore, Style

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

parent_root = os.path.dirname(os.path.abspath(__file__))
# 基础配置
cache_dir = "/home/myx/.cache/huggingface/hub"
model_name = "Qwen/Qwen2.5-7B-Instruct"
dataset_dir = os.path.join(parent_root,"GenAI-Hw5","Tang_training_data.json")

# 训练数据和输出配置
num_train_data = 1040  # 训练数据量 (最大值5000)
output_dir = os.path.join(parent_root,"output")  # 输出目录
ckpt_dir = os.path.join(parent_root,"exp1")     # checkpoint保存目录

# 训练超参数
num_epoch = 1           # 训练轮数
LEARNING_RATE = 3e-4    # 学习率
MICRO_BATCH_SIZE = 4    # 微批次大小
BATCH_SIZE = 16        # 批次大小
CUTOFF_LEN = 256       # 文本截断长度

# LoRA 参数配置
LORA_R = 8             # LoRA 的 R 值
LORA_ALPHA = 16        # LoRA 的 Alpha 值
LORA_DROPOUT = 0.05    # LoRA 的 Dropout 率
TARGET_MODULES = [
    "q_proj", "up_proj", "o_proj", 
    "k_proj", "down_proj", "gate_proj", "v_proj"
]

# 训练过程配置
logging_steps = 20      # 日志输出间隔
save_steps = 65        # 模型保存间隔
save_total_limit = 3   # 最多保存的checkpoint数量
VAL_SET_SIZE = 0       # 验证集大小
report_to = "none"     # 实验指标上报配置

# 设备和分布式训练配置
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size


def evaluate(instruction, generation_config, max_len, input_text="", verbose=True):
    """
    获取模型在给定输入下的生成结果。

    参数：
    - instruction: 描述任务的字符串。
    - generation_config: 模型生成配置。
    - max_len: 最大生成长度。
    - input_text: 输入文本，默认为空字符串。
    - verbose: 是否打印生成结果。

    返回：
    - output: 模型生成的文本。
    """
    # 构建完整的输入提示词
    prompt = f"""\
                [INST] <<SYS>>
                You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
                <</SYS>>

                {instruction}
                {input_text}
                [/INST]
            """

    # 将提示词转换为模型所需的 token 格式
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # 使用模型生成回复
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,               # 返回结构化成果
        output_scores=True,                         # 返回每个生成步骤的概率分数
        max_new_tokens=max_len,
    )

    # 解码并打印生成的回复
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        output = (
            output.split("[/INST]")[1]
            .replace("</s>", "")
            .replace("<s>", "")
            .replace("<|endoftext|>", "")  # 添加这行
            .replace("Assistant:", "")
            .replace("Assistant", "")
            .strip()
        )
        if verbose:
            print(output)

    return output


''' 选择需要微调的模型 '''
# 查找所有可用的 checkpoints
ckpts = []
for ckpt in os.listdir(ckpt_dir):
    if ckpt.startswith("checkpoint-"):
        ckpts.append(ckpt)

# 列出所有的 checkpoints
ckpts = sorted(ckpts, key=lambda ckpt: int(ckpt.split("-")[-1]))
print("所有可用的 checkpoints：")
print(" id: checkpoint 名称")
for (i, ckpt) in enumerate(ckpts):
    print(f"{i:>3}: {ckpt}")

id_of_ckpt_to_use = -1              # 要用于推理的 checkpoint 的 id（对应上一单元格的输出结果）。
                                    # 默认值 -1 表示使用列出的最后一个 checkpoint。
                                    # 如果你想选择其他 checkpoint，可以将 -1 更改为列出的 checkpoint id 中的某一个。

ckpt_name = os.path.join(ckpt_dir, ckpts[id_of_ckpt_to_use])
max_len = 128                       # 生成回复的最大长度
temperature = 0.1                   # 设置生成回复的随机度，值越小生成的回复越稳定。
top_p = 0.3                         # Top-p (nucleus) 采样的概率阈值，用于控制生成回复的多样性。
# top_k = 5                         # 调整 Top-k 值，以增加生成回复的多样性并避免生成重复的词汇。



''' 加载模型和分词器 '''
test_data_path = os.path.join(parent_root,"GenAI-Hw5","Tang_testing_data.json")  # 测试数据集的路径
output_path = os.path.join(output_dir, "results.txt")  # 生成结果的输出路径

seed = 42
no_repeat_ngram_size = 3

# 配置模型的量化设置
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    device_map={'': 0},
    cache_dir=cache_dir
)

# 加载微调后的权重
model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})


''' 生成测试结果 '''
results = []

# 设置生成配置，包括随机度、束搜索等参数
generation_config = GenerationConfig(
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    # top_k=top_k,  # 如果需要使用 top-k，可以在此设置
    no_repeat_ngram_size=no_repeat_ngram_size,
    pad_token_id=2,
)

# 读取测试数据集
with open(test_data_path, "r", encoding="utf-8") as f:
    test_datas = json.load(f)

# 对每个测试样例生成预测，并保存结果
with open(output_path, "w", encoding="utf-8") as f:
    for (i, test_data) in enumerate(test_datas):
        predict = evaluate(test_data["instruction"], generation_config, max_len, test_data["input"], verbose=False)
        f.write(f"{i+1}. " + test_data["input"] + predict + "\n")
        print(f"{i+1}. " + test_data["input"] + predict)

""" 微调前后对比 """

# 使用之前的测试例子
test_tang_list = [
    '相見時難別亦難，東風無力百花殘。',
    '重帷深下莫愁堂，臥後清宵細細長。',
    '芳辰追逸趣，禁苑信多奇。'
]

# 使用微调后的模型进行推理
demo_after_finetune = []
for tang in test_tang_list:
    demo_after_finetune.append(
        f'模型輸入:\n以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。{tang}\n\n模型輸出:\n' +
        evaluate('以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。', generation_config, max_len, tang, verbose=False)
    )

# 打印输出结果
for idx in range(len(demo_after_finetune)):
    print(f"Example {idx + 1}:")
    print(demo_after_finetune[idx])
    print("-" * 80)