import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


IGNORE_INDEX = -100


def build_instruction_prompt(instruction: str):
    """构建指令提示模板，针对C代码到汇编代码的转换任务"""
    return """<s>[INST] Given the following C code, generate the corresponding assembly code:

{}

[/INST]""".format(instruction.strip())


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="codellama/CodeLlama-34b-hf"
    )
    use_flash_attention: bool = field(
        default=False, metadata={"help": "Whether to use flash attention."}
    )
    # LoRA相关参数
    use_lora: bool = field(
        default=True, metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning."}
    )
    lora_r: int = field(
        default=16, metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # LoRA训练可以使用更大的batch size和学习率
    per_device_train_batch_size: int = field(default=4)  # LoRA可以使用更大的batch size
    gradient_accumulation_steps: int = field(default=4)  # 相应减少梯度累积步数
    learning_rate: float = field(default=2e-4)  # LoRA可以使用更大的学习率
    warmup_steps: int = field(default=100)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    logging_steps: int = field(default=50)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    # LoRA训练相关设置
    remove_unused_columns: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train_tokenize_function(examples, tokenizer):
    """处理训练数据的tokenization"""
    sources = [
        build_instruction_prompt(instruction) for instruction in examples["instruction"]
    ]
    eos_token = tokenizer.eos_token
    targets = [f"{output}</s>" for output in examples["output"]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def setup_lora_model(model, model_args):
    """设置LoRA模型"""
    if not model_args.use_lora:
        return model
    
    # 解析target modules
    target_modules = [module.strip() for module in model_args.lora_target_modules.split(",")]
    
    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias="none",  # 不训练bias
        modules_to_save=None,  # 不保存其他模块
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def create_deepspeed_config(training_args):
    """创建DeepSpeed配置，使用auto值来避免参数冲突"""
    config = {
        "fp16": {
            "enabled": training_args.fp16
        },
        "bf16": {
            "enabled": training_args.bf16
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "cpu_offload": False
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",  # 使用auto让HF自动设置
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto", 
                "warmup_num_steps": "auto",
                "total_num_steps": "auto"  # 关键：使用auto避免冲突
            }
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": training_args.logging_steps,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }
    
    return config


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 检查是否使用DeepSpeed
    if training_args.deepspeed:
        import json
        # 如果传入的是字符串路径，读取配置文件
        if isinstance(training_args.deepspeed, str):
            with open(training_args.deepspeed, 'r') as f:
                deepspeed_config = json.load(f)
        else:
            deepspeed_config = training_args.deepspeed
            
        # 更新DeepSpeed配置，使用auto值
        auto_config = create_deepspeed_config(training_args)
        
        # 将auto配置写入临时文件
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(auto_config, f, indent=2)
            temp_config_path = f.name
        
        # 更新training_args中的deepspeed配置路径
        training_args.deepspeed = temp_config_path
        
        if training_args.local_rank == 0:
            print("Updated DeepSpeed config with auto values:")
            print(json.dumps(auto_config, indent=2))

    if training_args.local_rank == 0:
        print("=" * 100)
        print(training_args)

    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # Code Llama的特殊token设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    # 模型加载配置
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    
    # 当使用DeepSpeed时，不要使用device_map="auto"
    if not training_args.deepspeed and not model_args.use_lora:
        model_kwargs["device_map"] = "auto"
    
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # 加载基础模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        **model_kwargs
    )

    # 应用LoRA（如果启用）
    model = setup_lora_model(model, model_args)

    # 启用梯度检查点以节省内存
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        # 对于LoRA模型，需要手动设置
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.local_rank == 0:
        print("Load model from {} over.".format(model_args.model_name_or_path))
        if model_args.use_lora:
            print("LoRA enabled with r={}, alpha={}, dropout={}".format(
                model_args.lora_r, model_args.lora_alpha, model_args.lora_dropout
            ))

    # 加载数据集
    raw_train_datasets = load_dataset(
        "json",
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir,
    )
    if training_args.local_rank > 0:
        torch.distributed.barrier()

    # 数据预处理
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=16,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer},
    )

    if training_args.local_rank == 0:
        torch.distributed.barrier()

    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(
                f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}."
            )
            print(
                f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}."
            )

    # 数据整理器
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

    # 创建训练器
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # 开始训练
    try:
        trainer.train()
    finally:
        # 清理临时DeepSpeed配置文件
        if training_args.deepspeed and hasattr(training_args, '_temp_deepspeed_config'):
            try:
                os.unlink(training_args._temp_deepspeed_config)
            except:
                pass
    
    # 保存模型
    if model_args.use_lora:
        # 保存LoRA权重
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model()
    
    trainer.save_state()


if __name__ == "__main__":
    train()