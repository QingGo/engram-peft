import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram_peft import EngramConfig, get_engram_model
from torchinfo import summary

# 1. 加载模型
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 2. 配置 Engram (使用一个小的配置以便可视化)
config = EngramConfig(
    target_layers=[2, 11],  # 注入到第 2 层和第 11 层
    hidden_size=2048,
    embedding_dim=1280,
    tokenizer_name_or_path=MODEL_NAME,
)

# 3. 注入 Engram
model = get_engram_model(base_model, config, tokenizer)

# 4. 使用 torchinfo 可视化
# input_size 的格式为 (batch_size, seq_len)
# 我们需要传入 token 类型的输入 (torch.long)
batch_size = 1
seq_len = 128
dummy_input = torch.zeros((batch_size, seq_len), dtype=torch.long)

print("\n" + "=" * 20 + " 模型结构总览 " + "=" * 20)
summary(
    model,
    input_data=dummy_input,
    depth=4,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    row_settings=["var_names"],
)
