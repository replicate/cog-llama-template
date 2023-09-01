from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# model_path = '../weights/codellama-7b-instruct/gptq'
model_path = '../weights/codellama-13b-instruct/gptq'
# model_path = '../weights/codellama-34b-instruct/gptq'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_path,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=False,
        quantize_config=None,
        disable_exllama=True,
        inject_fused_attention=False
)

prompt_template=f'''[INST] <<SYS>>
All responses must be written in TypeScript.
<</SYS>>

Write a function that sums 2 integers together and returns the results.
[/INST]
'''

# print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()

inference_kwargs = dict(inputs=input_ids, temperature=0.7, max_new_tokens=128, top_p=0.95, top_k=50, do_sample=True)
# print('#### inference params ####')
# for k,v in inference_kwargs.items():
#     print(f'{k}: {v}')
# print('########')
output = model.generate(**inference_kwargs)
print(tokenizer.decode(output[0]))
