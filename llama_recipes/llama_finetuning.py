# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import fire
import torch

# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
)
import torch.distributed as dist

# Unused imports removed
from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    print_model_size,
    get_policies,
)

from utils.dataset_utils import get_preprocessed_dataset

from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from peft import (
    get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.utils.data import DistributedSampler
import policies
from policies import AnyPrecisionAdamW
from configs import fsdp_config, train_config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def main(**kwargs):
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    #########################################################
    # CONFIGURE DISTRIBUTED TRAINING -----------------------
    #########################################################
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        import os

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
        setup_environ_flags(rank)

    #########################################################
    # INITIALIZE TOKENIZEER --------------------------------
    #########################################################
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name, legacy=False)

    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
    #########################################################
    # PREPARE TRAIN AND VALIDATION DATA --------------------
    #########################################################
    dataset_config = generate_dataset_config(train_config, kwargs)
    update_config(
        dataset_config,
        **{
            "data_path": train_config.data_path,
            "num_validation_samples": train_config.num_validation_samples,
            "validation_data_path": train_config.validation_data_path,
            "run_validation": train_config.run_validation,
            "pack_sequences": train_config.pack_sequences,
            "wrap_packed_sequences": train_config.wrap_packed_sequences,
            "chunk_size": train_config.chunk_size,
        },
    )

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    if train_config.run_validation:
        dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="val",
        )
        if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")
    else:
        dataset_val = None

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )

    # Create DataLoaders for the training and validation dataset
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding="longest"
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=data_collator,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=data_collator,
        )
    else:
        eval_dataloader = None

    if len(train_dataloader) == 0:
        raise ValueError(
            "Training dataloader is empty! This happens when your dataset is too small, relative to your batch size. "
            "If `pack_sequences` is `True`, you're more likely to run into this issue, particularly with small datasets that "
            "consist of short examples. Try setting `pack_sequences` to `False` and/or reducing your batch size."
        )

    #########################################################
    # CONFIGURE AND INITIALIZE MODEL ------------------------
    #########################################################

    # Model preparation for full fine-tuning -------
    # ----------------------------------------------
    if not train_config.use_peft:
        print("Loading model for peft")
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
        print("Loaded model")

    else:
        kwargs["r"] = kwargs[
            "lora_rank"
        ]  # can't pass --r to the script, torchrun won't have it
        peft_config = generate_peft_config(train_config.peft_method, kwargs)

        # Model preparation for QLoRA fine-tuning ------
        # ----------------------------------------------
        if train_config.peft_method == "qlora":
            print("LOADING MODEL FOR QLORA")
            bnb_config = generate_peft_config("bitsandbytes_config", kwargs)
            import os

            print(
                f"Loading model from {train_config.model_name}, which contains the following files:"
            )
            print(os.listdir(train_config.model_name))

            model = AutoModelForCausalLM.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                device_map="auto",  # dispatch efficiently the model on the available ressources
                # max_memory = {i: max_memory for i in range(num_gpus)},
            )
            print("Loaded model")

            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)

        # Model preparation for LoRA fine-tuning ------
        # ----------------------------------------------

        else:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )

        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # We added a special token for padding, so we need to resize the token embeddings
    model.resize_token_embeddings(model.config.vocab_size + 1)

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy
            if train_config.use_peft
            else wrapping_policy,
            mixed_precision=mixed_precision_policy
            if not fsdp_config.pure_bf16
            else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)

    # Note: When we use QLoRA, we load directly to devices with `automap`, so we don't need to move to cuda here.
    elif (
        not train_config.quantization
        and not train_config.enable_fsdp
        and not train_config.peft_method == "qlora"
    ):
        model.to("cuda")

    # Initialize the optimizer and learning rate scheduler
    if not train_config.peft_method == "qlora":
        if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
            optimizer = AnyPrecisionAdamW(
                model.parameters(),
                lr=train_config.lr,
                momentum_dtype=torch.bfloat16,
                variance_dtype=torch.bfloat16,
                use_kahan_summation=False,
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=train_config.lr,
                weight_decay=0.0,
            )

    gradient_accumulation_steps = train_config.gradient_accumulation_steps

    if not train_config.peft_method == "qlora":
        scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

        # Start the training process
        results = train(
            model,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            optimizer,
            scheduler,
            gradient_accumulation_steps,
            train_config,
            fsdp_config if train_config.enable_fsdp else None,
            local_rank if train_config.enable_fsdp else None,
            rank if train_config.enable_fsdp else None,
        )
        if not train_config.enable_fsdp or rank == 0:
            [print(f"Key: {k}, Value: {v}") for k, v in results.items()]

    else:
        from transformers import TrainingArguments, Trainer
        from trl.trainer.utils import PeftSavingCallback

        training_args = TrainingArguments(
            output_dir=train_config.output_dir,
            per_device_train_batch_size=train_config.batch_size_training,
            per_device_eval_batch_size=train_config.val_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=train_config.lr,
            bf16=True,
            log_level="info",
            logging_steps=10,
            optim="paged_adamw_32bit",
            warmup_ratio=0.03,
            save_strategy="no",
            num_train_epochs=train_config.num_epochs,
            gradient_checkpointing=True,
            do_eval=True,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            data_collator=data_collator,
            # peft_config=peft_config,
            args=training_args,
            compute_metrics=None,
            callbacks=[PeftSavingCallback],
        )

        trainer.train()

        trainer.model.save_pretrained(train_config.output_dir)


if __name__ == "__main__":
    fire.Fire(main)
