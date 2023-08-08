import json
import tqdm

if __name__ == "__main__":
    data = json.load(open('train_sampled.json', 'r'))
    for item in tqdm.tqdm(data):
        id1 = item['id1'].split('/')[1]+'_'+item['id1'].split('/')[-1]
        code1 = item['code1']
        with open('code/'+id1, 'w') as fp:
            fp.write(code1)
        id2 = item['id2'].split('/')[1]+'_'+item['id2'].split('/')[-1]
        code2 = item['code2']
        with open('code/'+id2, 'w') as fp:
            fp.write(code2)
    
    data = json.load(open('val_sampled.json', 'r'))
    for item in tqdm.tqdm(data):
        id1 = item['id1'].split('/')[1]+'_'+item['id1'].split('/')[-1]
        code1 = item['code1']
        with open('code/'+id1, 'w') as fp:
            fp.write(code1)
        id2 = item['id2'].split('/')[1]+'_'+item['id2'].split('/')[-1]
        code2 = item['code2']
        with open('code/'+id2, 'w') as fp:
            fp.write(code2)
    
    data = json.load(open('test_sampled.json', 'r'))
    for item in tqdm.tqdm(data):
        id1 = item['id1'].split('/')[1]+'_'+item['id1'].split('/')[-1]
        code1 = item['code1']
        with open('code/'+id1, 'w') as fp:
            fp.write(code1)
        id2 = item['id2'].split('/')[1]+'_'+item['id2'].split('/')[-1]
        code2 = item['code2']
        with open('code/'+id2, 'w') as fp:
            fp.write(code2)