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
tmp_dataset_dir = os.path.join(parent_root,"GenAI-Hw5","tmp_dataset.json")

# 训练数据和输出配置
num_train_data = 1040  # 训练数据量 (最大值5000)
output_dir = os.path.join(parent_root,"output")  # 输出目录
ckpt_dir = os.path.join(parent_root,"exp1")     # checkpoint保存目录

# 训练超参数
num_epoch = 3           # 训练轮数
LEARNING_RATE = 3e-4    # 学习率
MICRO_BATCH_SIZE = 4    # 微批次大小
BATCH_SIZE = 16        # 批次大小
CUTOFF_LEN = 256       # 文本截断长度

# LoRA 参数配置
LORA_R = 8             # LoRA 的 R 值
LORA_ALPHA = 16        # LoRA 的 Alpha 值
LORA_DROPOUT = 0.05    # LoRA 的 Dropout 率
TARGET_MODULES = [                                  # 指定哪些层的权重添加LoRA适配器（如注意力层的q_proj, v_proj）
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


# 设置量化方式
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                         # 核心开关：启用4位量化加载模型
    bnb_4bit_quant_type="nf4",                 # 量化算法：使用NF4（NormalFloat4）格式，专为神经网络权重优化
    bnb_4bit_compute_dtype=torch.float16,      # 计算精度：前向传播和反向传播时用FP16加速计算
    bnb_4bit_use_double_quant=False,           # 双重量化：禁用（若为True，可进一步压缩量化参数，但增加计算开销）
)

# 修改模型加载部分
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,            # 关键：应用上述量化配置
    torch_dtype=torch.float16,                 # 非量化部分（如某些层）用FP16加载
    device_map=device_map,                     # 自动分配模型层到GPU/CPU（如`"auto"`）
    cache_dir=cache_dir,                       # 指定模型缓存目录（避免重复下载）
)

# 创建 tokenizer 并设置结束符号 (eos_token)
logging.getLogger("transformers").setLevel(logging.ERROR)       #限制只有ERROR才会在日志中输出
tokenizer = AutoTokenizer.from_pretrained(      #分词器，转换文本、添加特殊标记、指定模型缓存目录
    model_name,                                 #模型名称
    add_eos_token=True,                         #给文本添加终止符（默认为False）
    cache_dir=cache_dir
)
tokenizer.pad_token = tokenizer.eos_token       #设置填充标记与结束标记相同，确保输入序列具有相同长度

# 设置模型推理时的解码参数
max_len = 128                                   #生成文本的最大长度
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    num_beams=1,
    top_p=0.3,                                  
    no_repeat_ngram_size=3,                     #避免重复3个词的组合
    pad_token_id=2,                             #设置填充标记的ID
)



def generate_training_data(data_point):
    """
    将输入和输出文本转换为模型可读取的 tokens。

    参数：
    - data_point: 包含 "instruction"、"input" 和 "output" 字段的字典。

    返回：
    - 包含模型输入 IDs、标签和注意力掩码的字典。

    示例:
    - 如果你构建了一个字典 data_point_1，并包含字段 "instruction"、"input" 和 "output"，你可以像这样使用函数：
        generate_training_data(data_point_1)
    """
    # 构建完整的输入提示词          instruction：告诉模型要做什么        input：告诉模型对什么做
    prompt = f"""\
                [INST] <<SYS>>
                You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
                <</SYS>>

                {data_point["instruction"]}
                {data_point["input"]}
                [/INST]
            """

    # 计算用户提示词的 token 数量
    len_user_prompt_tokens = (
        len(
            tokenizer(            #对文本进行编码，将文本转化为数字token ID， 输出input_ids和attention_mask列表
                prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
                padding="max_length",
            )["input_ids"]
        )
        - 1
    )

    # 将完整的输入和输出转换为 tokens
    full_tokens = tokenizer(
        prompt + " " + data_point["output"] + "</s>",           # 设置需要转换的内容
        truncation=True,                                        # 启用截断
        max_length=CUTOFF_LEN + 1,                              # 最大长度控制
        padding="max_length",                                   # 如果文本太短，填充到固定长度
    )["input_ids"][:-1]

    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * len(full_tokens),
    }

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
            .replace("</s>", "")                # 句子开始标记
            .replace("<s>", "")                 # 句子结束标记
            .replace("<|endoftext|>", "")       # 添加这行，文本结束标记
            .replace("Assistant:", "")
            .replace("Assistant", "")
            .strip()                            # 去除首尾空白字符串
        )
        if verbose:
            print(output)

    return output


# 测试样例
test_tang_list = [
    "相見時難別亦難，東風無力百花殘。",
    "重帷深下莫愁堂，臥後清宵細細長。",
    "芳辰追逸趣，禁苑信多奇。",
]

# 获取每个样例的模型输出
demo_before_finetune = []
for tang in test_tang_list:
    demo_before_finetune.append(
        f"模型輸入:\n以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。{tang}\n\n模型輸出:\n"
        + evaluate(
            "以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。",
            generation_config,
            max_len,
            tang,
            verbose=False,
        )
    )

# 打印并将输出存储到文本文件
for idx in range(len(demo_before_finetune)):
    print(f"Example {idx + 1}:")
    print(demo_before_finetune[idx])
    print("-" * 80)



""" 训练部分 """
print("--------------训练部分开始--------------\n")
# 设置TOKENIZERS_PARALLELISM为false，这里简单禁用并行性以避免报错
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 创建指定的输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# 对量化模型进行预处理以进行训练
model = prepare_model_for_kbit_training(model)      #冻结主参数 + 解决梯度问题 + 优化显存​​

# 使用 LoraConfig 配置 LORA 模型
config = LoraConfig(
    r=LORA_R,                                       # 秩（Rank）：控制LoRA适配器的“宽度”，决定参数数量
    lora_alpha=LORA_ALPHA,                          # 缩放系数：控制LoRA权重对原始权重的调整强度
    target_modules=TARGET_MODULES,                  # 目标模块：指定哪些层的权重添加LoRA适配器（如注意力层的q_proj, v_proj）
    lora_dropout=LORA_DROPOUT,                      # Dropout率：防止LoRA层过拟合，通常在0~0.5之间
    bias="none",                                    # 偏置项处理：不训练原始模型的偏置参数（可选值："none"/"all"/"lora_only"）
    task_type="CAUSAL_LM",                          # 任务类型：指定为因果语言模型（如GPT、LLaMA的预训练任务）
)
model = get_peft_model(model, config)

# 将 tokenizer 的填充 token 设置为 结束id
tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载并处理训练数据
with open(dataset_dir, "r", encoding="utf-8") as f:
    data_json = json.load(f)
with open(tmp_dataset_dir, "w", encoding="utf-8") as f:
    json.dump(data_json[:num_train_data], f, indent=2, ensure_ascii=False)

data = load_dataset('json', data_files=tmp_dataset_dir, download_mode="force_redownload")

# 将训练数据分为训练集和验证集（若 VAL_SET_SIZE 大于 0）
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_training_data)
    val_data = train_val["test"].shuffle().map(generate_training_data)
else:
    train_data = data['train'].shuffle().map(generate_training_data)
    val_data = None


# 使用 Transformers Trainer 进行模型训练
trainer = transformers.Trainer(
    # 参数1：要训练的模型（已加载或添加了LoRA）
    model=model,  
    # 参数2：训练数据集
    train_dataset=train_data,
    # 参数3：验证数据集（可选）
    eval_dataset=val_data,
    # 参数4：训练超参数配置（核心部分）
    args=transformers.TrainingArguments(
        # ---- 批次与梯度配置 ----
        per_device_train_batch_size=MICRO_BATCH_SIZE,                   # 每个GPU的批次大小（如1~4，取决于显存）
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,        # 梯度累积步数（模拟大批次）
        # ---- 学习率与优化 ----
        warmup_steps=50,                    # 学习率预热步数（逐渐增大学习率避免初期震荡）
        num_train_epochs=num_epoch,         # 训练总轮次
        learning_rate=LEARNING_RATE,        # 初始学习率（LoRA常用1e-4 ~ 5e-5）
        # ---- 精度与速度 ----
        fp16=True,                          # 启用混合精度训练（利用GPU TensorCore加速）
        # ---- 日志与保存 ----
        logging_steps=logging_steps,        # 每隔多少步记录日志（如50步）
        save_strategy="steps",              # 保存策略：按步数保存（可选"epoch"按轮次）
        save_steps=save_steps,              # 每隔多少步保存模型（如200步）
        output_dir=ckpt_dir,                # 模型和日志保存路径
        save_total_limit=save_total_limit,  # 最多保留的检查点数量（避免占满磁盘）
        # ---- 分布式训练 ----
        ddp_find_unused_parameters=False if ddp else None,  # DDP模式中是否检测未使用参数（LoRA需关闭）
        # ---- 监控与报告 ----
        report_to=report_to,                                # 日志上报工具（如"tensorboard"、"wandb"）
    ),
    # 参数5：数据整理器（处理数据成模型输入格式）
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer,  # 分词器（与模型匹配）
        mlm=False   # 是否使用遮蔽语言模型（MLM），False表示因果语言建模（如GPT）
    ),
)

# 禁用模型的缓存功能
model.config.use_cache = False

# 若使用 PyTorch 2.0 以上版本且非 Windows 系统，编译模型
if torch.__version__ >= "2" and sys.platform != 'win32':
    model = torch.compile(model)

# 开始模型训练
trainer.train()

# 将训练好的模型保存到指定目录
model.save_pretrained(ckpt_dir)

# 打印训练过程中可能出现的缺失权重警告信息
print("\n 如果上方有关于缺少键的警告，请忽略 :)")