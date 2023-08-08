import json
import os
import tqdm

js_all = []


"""处理数据，使格式符合ALERT"""
with open('data.jsonl', 'w') as f:
    with open('train_sampled.txt', 'w') as fp:
        train_data = json.load(open('train_sampled.json'))
        for item in tqdm.tqdm(train_data):
            id1 = item['id1']
            code1 = item['code1']
            id2 = item['id2']
            code2 = item['code2']
            label = item['label']
            fp.write(id1+'\t'+id2+'\t'+str(label)+'\n')
            if id1 not in js_all:
                js_all.append(id1)
                js = {}
                js['idx'] = id1
                js['func'] = code1
                f.write(json.dumps(js)+'\n')
            if id2 not in js_all:
                js_all.append(id2)
                js = {}
                js['idx'] = id2
                js['func'] = code2
                f.write(json.dumps(js)+'\n')

    with open('val_sampled.txt', 'w') as fp:
        val_data = json.load(open('val_sampled.json'))
        for item in tqdm.tqdm(val_data):
            id1 = item['id1']
            code1 = item['code1']
            id2 = item['id2']
            code2 = item['code2']
            label = item['label']
            fp.write(id1+'\t'+id2+'\t'+str(label)+'\n')
            if id1 not in js_all:
                js_all.append(id1)
                js = {}
                js['idx'] = id1
                js['func'] = code1
                f.write(json.dumps(js)+'\n')
            if id2 not in js_all:
                js_all.append(id2)
                js = {}
                js['idx'] = id2
                js['func'] = code2
                f.write(json.dumps(js)+'\n')

    with open('test_sampled.txt', 'w') as fp:
        test_data = json.load(open('test_sampled.json'))
        for item in tqdm.tqdm(test_data):
            id1 = item['id1']
            code1 = item['code1']
            id2 = item['id2']
            code2 = item['code2']
            label = item['label']
            fp.write(id1+'\t'+id2+'\t'+str(label)+'\n')
            if id1 not in js_all:
                js_all.append(id1)
                js = {}
                js['idx'] = id1
                js['func'] = code1
                f.write(json.dumps(js)+'\n')
            if id2 not in js_all:
                js_all.append(id2)
                js = {}
                js['idx'] = id2
                js['func'] = code2
                f.write(json.dumps(js)+'\n')
