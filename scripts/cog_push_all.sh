#!/bin/bash

model_names=("llama-7b")

for model_name in "${model_names[@]}"; do
  echo "Pushing model: $model_name"
  cog run python select_model.py --model_name $model_name
  cog login --token-stdin <<< "$COG_TOKEN"
  cog push r8.im/replicate/$model_name
done



