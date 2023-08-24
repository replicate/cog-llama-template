#!/bin/bash

# TODO - rework this to spin up cog servers locally for prediction & training
# this gives us the ability to test out post-training results (w/docker "env" vars)
# I think that'll actually do it. 

cog predict -i prompt="Hey! How are you doing?"
cog train -i train_data="https://storage.googleapis.com/dan-scratch-public/fine-tuning/1k_samples_prompt.jsonl" -i max_steps=10
