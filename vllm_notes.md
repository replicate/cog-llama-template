**python vllm/benchmarks/benchmark_latency.py --model llama_weights/llama-2-70b/ --batch-size 1 -tp 2**

```
#Output 
Namespace(model='llama_weights/llama-2-70b/', tokenizer=None, tensor_parallel_size=2, input_len=32, output_len=128, batch_size=1, n=1, use_beam_search=False, num_iters=3, trust_remote_code=False)

Initializing an LLM engine with config: model='llama_weights/llama-2-70b/', tokenizer='llama_weights/llama-2-70b/', tokenizer_mode=auto, trust_remote_code=False, dtype=torch.float16, use_dummy_weights=False, download_dir=None, use_np_weights=False, tensor_parallel_size=2, seed=0)

SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=1.0, top_p=0.9, top_k=-1, use_beam_search=False, stop=[], ignore_eos=True, max_tokens=128, logprobs=None)

Avg latency: 7.503512382507324 seconds
GPU : 75112MiB

t/s = 1000/(avg_latency*1000/128) = 17.05867778647263
```

*BS=2*

```python vllm/benchmarks/benchmark_latency.py --model llama_weights/llama-2-70b/ --batch-size 2 -tp 2```

```
Namespace(model='llama_weights/llama-2-70b/', tokenizer=None, tensor_parallel_size=2, input_len=32, output_len=128, batch_size=2, n=1, use_beam_search=False, num_iters=3, trust_remote_code=False)

SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=1.0, top_p=0.9, top_k=-1, use_beam_search=False, stop=[], ignore_eos=True, max_tokens=128, logprobs=None)

output_len = 128
Avg latency: 7.818761984507243 seconds

max GPU ~ 75118MiB x 2

t/s = 1000/(avg_latency*1000/128) = 16.3

```

*BS=2, output_len=4000*

**NOTE: I REQUESTED OUTPUT LENGTH OF 4000, BUT ONLY GOT 2017**

```python vllm/benchmarks/benchmark_latency.py --model llama_weights/llama-2-70b/ --batch-size 2 -tp 2 --output-len 4000```

```
Namespace(model='llama_weights/llama-2-70b/', tokenizer=None, tensor_parallel_size=2, input_len=32, output_len=4000, batch_size=2, n=1, use_beam_search=False, num_iters=3, trust_remote_code=False)

SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=1.0, top_p=0.9, top_k=-1, use_beam_search=False, stop=[], ignore_eos=True, max_tokens=4000, logprobs=None)

output_len = 2017
Avg latency: 139.3498539129893 seconds

max GPU ~ 75118MiB x 2

t/s = 1000/(avg_latency*1000/128) = 14.47
```

**bs=4, output_len=128**

```
python vllm/benchmarks/benchmark_latency.py --model llama_weights/llama-2-70b/ --batch-size 4 -tp 2 --output-len 128 

~GPU USAGE = 75148MB

avg_latency = 7.960791269938151 seconds
tps = 16.078803684171266
```

**bs=8, output_len=128**

```
python vllm/benchmarks/benchmark_latency.py --model llama_weights/llama-2-70b/ --batch-size 8 -tp 2 --output-len 128 

~GPU USAGE = 75152MiB

avg_latency = 8.235888799031576 seconds
tps = 15.54173485380849
throughput = 124
```


**bs=16, output_len=128**

```
python vllm/benchmarks/benchmark_latency.py --model llama_weights/llama-2-70b/ --batch-size 16 -tp 2 --output-len 128 

~GPU USAGE = 75204MiB

avg_latency = 9.124814828236898 seconds
tps = 14.027681921160939
throughput = 224
```

**bs=32, output_len=128**

```
python vllm/benchmarks/benchmark_latency.py --model llama_weights/llama-2-70b/ --batch-size 32 -tp 2 --output-len 128 

~GPU USAGE = 75302MiB

avg_latency = 10.249749660491943
tps = 12.48810987973501
throughput = 399.6195161515203
```

**bs=64, output_len=128**

```
python vllm/benchmarks/benchmark_latency.py --model llama_weights/llama-2-70b/ --batch-size 64 -tp 2 --output-len 128 

~GPU USAGE = 75210MiB

avg_latency = 13.740530172983805
tps = 9.315506635374925
throughput = 596.1924246639952
```
