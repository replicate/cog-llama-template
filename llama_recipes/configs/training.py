# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str = "llama_weights/llama-2-7b"
    enable_fsdp: bool = False
    run_validation: bool = True
    batch_size_training: int = 4
    num_epochs: int = 3
    num_workers_dataloader: int = 1
    gradient_accumulation_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.85
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset = "completion"
    peft_method: str = "lora"  # None , llama_adapter, prefix
    use_peft: bool = False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str = (
        "PATH/to/save/FSDP/model"  # will be used if using FSDP
    )
    dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP
    save_optimizer: bool = False  # will be used if using FSDP
    data_path: str = None
    num_validation_samples: int = 100
    validation_data_path: str = None
    validation_prompt: str = None
    wrap_packed_sequences: bool = False
    pack_sequences: bool = True
    chunk_size: int = 2048

    # optim: Optional[str] = field(
    #     default="paged_adamw_32bit",
    #     metadata={"help": "The optimizer to use."},
    # )
    # lr_scheduler_type: str = field(
    #     default="constant",
    #     metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    # )
    # max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    # warmup_ratio

    # save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X updates steps."})
    # logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    # eval_steps: int = field(default=None, metadata={"help": "Run evaluation every X steps"})
    # evaluation_strateg
