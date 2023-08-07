import re
import os
from collections import Counter
import tqdm
import operator
import json
import subprocess
import itertools
from multiprocessing import Pool


def get_file_path(root_path, file_list):
    PATH = os.listdir(root_path)
    for path in PATH:
        co_path = os.path.join(root_path, path)
        if os.path.isfile(co_path):
            file_list.append(co_path)
        elif os.path.isdir(co_path):
            get_file_path(co_path, file_list)
    return file_list

def external_cmd(cmd, msg_in=''):
    try:
        proc = subprocess.Popen(cmd,
                                shell=True,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                )
        stdout_val, stderr_val = proc.communicate(msg_in)
        return stdout_val, stderr_val
    except ValueError as err:
        return None, None
    except IOError as err:
        return None, None

def get_tokens(data):
    """
    获取代码中对应的token
    """
    tokens=[]
    data=data.split('\n')
    for line in data:
        if(line.startswith("<") and (line.find("/>")!=-1)):
            id=line[line.index("text=")+6:line.index("\"/>")]
            tokens.append(id)  
    return tokens

def get_file_tokens(code_path):
    _, stderr_val = external_cmd("txl -q -Dscan " + code_path)
    cname = code_path.split("\\")[-1]
    code = bytes.decode(stderr_val)
    # print(cname)
    ctokens = get_tokens(code)
    return cname, ctokens


if __name__ == "__main__":
    # code_file_list = []
    # file_to_tokens = {}
    # get_file_path("../valid_code", code_file_list)
    # # print(code_file_list)
    # code_tokens = []

    # bar =  tqdm.tqdm(code_file_list)
    # for fc in bar:
    #     fname = fc.split("\\")[-1]
    #     fdata = open(fc, 'r').read()
    #     ftokens = re.sub(r'(\s)+', ' ', fdata).strip()
    #     ftokens = ftokens.split(' ')
    #     ftokens = list(Counter(ftokens))
    #     file_to_tokens[fname] = ftokens
    #     code_tokens.extend(ftokens)
    #     bar.update()
    # bar.close()

    # code_tokens = dict(Counter(code_tokens))
    # code_tokens = dict(sorted(code_tokens.items(), key=operator.itemgetter(1)))
    # with open("code_corpus.json", 'w') as fp:
    #     json.dump(code_tokens, fp)
    
    # with open("file_to_tokens.json", 'w') as fp:
    #     json.dump(file_to_tokens, fp)
    
    # print(len(code_tokens))

    # '''生成文件和代码对，后续会进行筛选'''
    # # code_tokens = json.load(open('code_corpus.json', 'r'))
    # # file_to_tokens =json.load(open('file_to_tokens.json', 'r'))
    # with open("temp_code_content.txt", 'w') as fp:
    #     start =160
    #     end = 170
    #     while end <= 400:  # &140_150_360 25&
    #         key_tokens = []
    #         # for i, item in enumerate(code_tokens.items()):
    #         #     if i > int(len(code_tokens) / 3) and i < int(len(code_tokens) / 3) + 200:
    #         #         key_tokens.append(item[0])
    #         for key, val in code_tokens.items():
    #             # if val >=40 and val <=50:
    #             #     count += 1
    #             if val >= start and val <end and key not in key_tokens:  #122,190
    #                 # if count % 43 == 0:
    #                 key_tokens.append(key)
            
    #         # with open('val_count.json', 'w') as fp:
    #         #     json.dump(val_count, fp)

    #         print(len(key_tokens))
    #         # with open("key_tokens.json", 'w') as fp:
    #         #     json.dump(key_tokens, fp)
    #         pbar = tqdm.tqdm(file_to_tokens.items())
    #         for na, to in pbar:
    #             cnt = 0
    #             tag = True
    #             label = ""
    #             for kt in key_tokens:
    #                 if kt in to:
    #                     cnt += 1
    #                     label = kt
    #                 if cnt > 1:
    #                     tag = False
    #                     break
    #             if tag == True and label != "":
    #                 fp.write(na + "  " + label + '\n')
    #             pbar.update()
    #         pbar.close()
    #         start += 10
    #         end += 10


    # file_tokens = {}
    # final_content = {}
    # with open("temp_code_content.txt", 'r') as fp:
    #     all_data =  fp.read().split('\n')
    #     for line in all_data:
    #         if line == "":
    #             continue
    #         file_name = line.split("  ")[0]
    #         token = line.split("  ")[1]
    #         if file_name not in file_tokens:
    #             file_tokens[file_name] = token
    # print(len(file_tokens))
    # val_count = {}
    # for k, v in file_tokens.items():
    #     if v not in val_count:
    #         val_count[v] = 1
    #     else:
    #         val_count[v] += 1
    # print(len(val_count))
    # type_cnt = []
    # for k, v in file_tokens.items():
    #     if val_count[v] >= 21:  # 60_70_25  **50_60_18 500 12632**  **60_70_19 496 14398**  **70_80_21 440 14525**  **80_90_22 407 15272**  **90_100_22 362 15146**
    #         final_content[k] = v
    #         if v not in type_cnt:
    #             type_cnt.append(v)

    # print(len(final_content))
    # print(len(type_cnt))
    # # with open("final_content.json", 'w') as fp:
    # #     json.dump(final_content, fp)

    # '''start=160, end=170, end<400, types=304, file_num=19927最终手动删掉最少的2类使得types=300'''
    # with open("code_content.txt", 'w') as fp:
    #     for k, v in final_content.items():
    #         if v == "failed\\n\");" or v == "-1,":
    #             continue
    #         fp.write(k + "  " + v + '\n')

    with open("code_content.txt", 'r') as fp:
        all_data = fp.read().split("\n")
        ftcnt = {}
        for line in all_data:
            if line == "":
                continue
            fname = line.split("  ")[0]
            token = line.split("  ")[1]
            if token not in ftcnt:
                ftcnt[token] = 1
            else:
                ftcnt[token] += 1
    # ftcnt = dict(sorted(ftcnt.items(), key=operator.itemgetter(1)))
    # with open("file_token_count.json", 'w') as fp:
    #     json.dump(ftcnt, fp)
    print(len(ftcnt))
    token_label = {}
    for i, item in enumerate(ftcnt.keys()):
        token_label[item] = i
    # print(token_label)
    with open("code_content_labels.txt", 'w') as fp:
        for line in all_data:
            if line == "":
                continue
            fname = line.split("  ")[0]
            token = line.split("  ")[1]
            fp.write(fname + '  ' + str(token_label[token]) + '\n')
        
