import json 

def load_data(fn):
    data = []
    with open(fn, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

def process_data(data):
    for i in data:
        i["id"] = i["seq"].split(" ")[0]
    return data

def count_ids(all_data):
    ids = {}
    for data in all_data:
        for i in data:
            _id = int(i["id"])
            if _id not in ids:
                ids[_id] = 1
            ids[_id] += 1
    
    return ids
    

if __name__ == '__main__':
    ranks = range(0,4)
    all_data = []
    for rank in ranks:
        fn = f'./bug/dataloader_rank_{rank}.jsonl'
        data = load_data(fn)
        data = process_data(data)
        all_data.append(data)
    
    assert len(all_data) == len(ranks)

    id_counts = count_ids(all_data)
    sorted_ids = sorted(id_counts.keys())

    
    missing_ids = []
    for i in list(range(1,301)):
        if i not in sorted_ids:
            missing_ids.append(i)
        else:
            print(f"Count for {i}: {id_counts[i]})")
    
    print(f"Missing ids: {missing_ids}")
    print(f"Length of missing ids: {len(missing_ids)}")
            