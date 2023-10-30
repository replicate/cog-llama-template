Make data

```
seq 1 300 | awk '{print "{\"text\": \"" $1 " : This is example " $1 " and I'\''ve added extra text just so it has a few tokens!\"}"}' > bug/data.jsonl
```


```
export SELECTED_MODEL=llama-2-7b
make select
cog build --no-cache

```

```
cog train -i train_data=@bug/data.json -i run_validation=False
```

for i in {0..3}; do wc -l "./bug/dataloader_rank_${i}.jsonl"; done


```
python bug/process_data_distribution.py 
```