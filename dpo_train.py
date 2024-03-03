import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig

from trl import DPOTrainer

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    # data parameters
    beta: Optional[float] = field(default=0.2, metadata={"help": "the beta parameter for DPO loss"})
    
    # dataset params
    dataset_path: Optional[str] = field(default="sahil2801/CodeAlpaca-20k", metadata={"help": "the path of the training dataset"})
    sample_percentage: Optional[int] = field(default=100, metadata={"help": "Percentage of element to sample"})
    
    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
        
    learning_rate: Optional[float] = field(default=2e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="constant", metadata={"help": "the lr scheduler type"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
        
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the maximum prompt length"})

    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )


def format_dataset(
    dataset: Dataset,
    cache_dir: Optional[str] = None,
    num_proc=24,
) -> Dataset:
    original_columns = dataset.column_names

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Instruct: " + question + "\n\nOutput: " for question in samples["prompt"]],
            "chosen": samples["selected"],
            "rejected": samples["rejected"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config = bnb_config,
    )
    
    model.config.use_cache = False

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config = bnb_config
    )
    
    model_ref.config.use_cache = False
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "microsoft/phi-2",
        model_max_length=script_args.max_length,
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(script_args.dataset_path, split = f"train[:{script_args.sample_percentage}%]")
    train_dataset = format_dataset(dataset)

    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        logging_steps=script_args.logging_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        num_train_epochs=script_args.num_train_epochs,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        optim=script_args.optimizer_type,
        fp16=False,
        remove_unused_columns=False,
    )    

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)