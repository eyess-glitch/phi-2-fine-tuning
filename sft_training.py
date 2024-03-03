from dataclasses import dataclass, field
from datasets import load_dataset

import torch
import torch.distributed
import transformers
from transformers import TrainingArguments, HfArgumentParser
from trl import SFTTrainer, ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config, DataCollatorForCompletionOnlyLM

@dataclass
class ScriptArguments:
    dataset_name: str = field(default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    sample_percentage: int = field(default=100, metadata={"help": "Percentage of element to sample"})

def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"Instruct: {example['instruction'][i]}\nOutput: {example['output'][i]}"
            output_texts.append(text)
        return output_texts
    
    
def train():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quantization_config = get_quantization_config(model_config)
    quantization_config.bnb_4bit_compute_dtype = torch.bfloat16

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        model_max_length=args.max_seq_length,
        padding_side="right",
        use_fast=True,
    )
    
    tokenizer.pad_token = tokenizer.eos_token
 
    train_dataset = load_dataset(args.dataset_name, split = f"train[:{args.sample_percentage}%]")
    response_template = "Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        model_init_kwargs=model_kwargs,
        max_seq_length=args.max_seq_length,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        formatting_func=formatting_prompts_func,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()