import json
import pandas as pd
import pickle as pkl
import tqdm


def get_jsonl_file(target_file, s_path):
    db = pd.read_pickle(s_path)
    with open(target_file, 'w') as f:
        for _, item in tqdm.tqdm(db.iterrows()):
            js = {}
            js['idx'] = item.id
            js['target'] = item.label
            js['func'] = item.func
            f.write(json.dumps(js)+'\n')
            

if __name__ == "__main__":
    train_path = '../data/train_set.pkl'
    val_path = '../data/val_set.pkl'
    test_path = '../data/test_set.pkl'

    get_jsonl_file('./dataset/train.jsonl', train_path)
    get_jsonl_file('./dataset/val.jsonl', val_path)
    get_jsonl_file('./dataset/test.jsonl', test_path)