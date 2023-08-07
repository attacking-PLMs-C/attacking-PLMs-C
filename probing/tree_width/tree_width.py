import os
import tqdm
import re
import operator
from collections import Counter

def get_file_path(root_path, file_list):
    PATH = os.listdir(root_path)
    for path in PATH:
        # print(path)
        co_path = os.path.join(root_path, path)
        if os.path.isfile(co_path):
            file_list.append(co_path)
        elif os.path.isdir(co_path):
            get_file_path(co_path, file_list)
    return file_list

if __name__ == "__main__":
    file_path_list = []
    get_file_path('../valid_code', file_path_list)
    tree_widths = {}
    bar = tqdm.tqdm(file_path_list)
    with open("tree_width.txt", 'w') as fp:
        with open("all_tree_depths.txt", 'w') as fp1:
            for path in bar:
                fname = path.split('/')[-1].split('.')[0]
                dot_path =  '../ast_dot/' + fname + '/1-ast.dot'
                if not os.path.exists(dot_path):
                    continue
                fdata = open(dot_path, 'r').read().split('\n')
                # print(fdata[1])
                # print(fdata[3])
                all_nodes ={}
                cur_depth = {}     #将每一个子节点深度设为1
                tag = True         #root节点标签
                root = ''
                tree_depth = 0
                for line in fdata:
        #             # print(line)
        #             # num = line.split('" [')[0].split('"')[-1]
        #             # pattern = re.compile(r'block_crypto_open_opts_init')
        #             # node_data = re.search(r'<.*</SUB>>', line)
        #             # if node_data != None and num != None:
        #             #    node_data = node_data.group()
        #             #    print(num)
        #             #    print(node_data)
        #             # print('-------------------------------')
                    edge_data = re.search(r'.*" -> ".*', line)
                    if edge_data != None:
                        edge_data  = edge_data.group().strip()
                        start = edge_data.split('"')[1]
                        end = edge_data.split('"')[-2]
                        # print(start + '  ' + end)
                        if tag == True:
                            root = start
                            tag = False
                        if start not in all_nodes:
                            all_nodes[start] = []
                        all_nodes[start].append(end)
                        if start not in cur_depth:
                            cur_depth[start] =  1
                        if end not in cur_depth:
                            cur_depth[end] = 1
                    # break
                # print(fname)
                # print(all_nodes)
                # print(root)
                # print(cur_depth)
                for node in all_nodes.keys():
                    for sub_node in all_nodes[node]:
                        cur_depth[sub_node] += cur_depth[node]
                        if tree_depth < cur_depth[sub_node]:
                            tree_depth  = cur_depth[sub_node]   #每个子节点深度等于“1+父节点所处深度”

                # print(tree_depth)
                fp1.write(fname + '.c  ' + str(tree_depth) + '\n')
                tree_w = 0
                cur_width = {}
                for i in range(tree_depth):
                    cur_width[str(i+1)] = 0
                cur_width['1'] = 1
                for node in all_nodes.keys():
                    cur_width[str(int(cur_depth[node]+1))] += len(all_nodes[node])  #子节点所在层数宽度+=node子节点个数
                for k in cur_width:
                    if tree_w < cur_width[k]:
                        tree_w = cur_width[k]
                # print(cur_width)
                # print(fname)
                # print(tree_w)
                # break
                fp.write(fname + '.c  ' + str(tree_w) + '\n')
                tree_widths[fname+'.c'] = tree_w
            bar.update()
    bar.close()

    with open("tree_width_labels.txt", 'w') as fp:
        for key, val in tree_widths.items():
            if val >= 3 and val < 7:
                fp.write(key + '  0\n')
            elif val >= 7 and val < 11:
                fp.write(key + '  1\n')
            elif val >= 11 and val < 15:
                fp.write(key + '  2\n')
            elif val >= 15 and val < 19:
                fp.write(key + '  3\n')
            elif val >= 19 and val < 23:
                fp.write(key + '  4\n')
            elif val >= 23 and val < 27:
                fp.write(key + '  5\n')
            elif val >= 27 and val < 31:
                fp.write(key + '  6\n')
            elif val >= 31 and val < 35:
                fp.write(key + '  7\n')
            elif val >= 35 and val < 39:
                fp.write(key + '  8\n')
            elif val >= 39 and val < 43:
                fp.write(key + '  9\n')
            elif val >= 43 and val < 47:
                fp.write(key + '  10\n')
            elif val >= 47 and val < 51:
                fp.write(key + '  11\n')
            elif val >= 51 and val < 55:
                fp.write(key + '  12\n')
            elif val >= 55 and val < 59:
                fp.write(key + '  13\n')
            elif val >= 59 and val < 63:
                fp.write(key + '  14\n')
            elif val >= 63 and val < 67:
                fp.write(key + '  15\n')
            elif val >= 67 and val < 71:
                fp.write(key + '  16\n')
            elif val >= 71 and val < 75:
                fp.write(key + '  17\n')
            elif val >= 75 and val < 79:
                fp.write(key + '  18\n')
            elif val >= 79 and val < 83:
                fp.write(key + '  19\n')
            elif val >= 83 and val < 87:
                fp.write(key + '  20\n')
            elif val >= 87 and val < 91:
                fp.write(key + '  21\n')
            elif val >= 91 and val < 95:
                fp.write(key + '  22\n')
            elif val >= 95 and val < 99:
                fp.write(key + '  23\n')
            elif val >= 99 and val < 103:
                fp.write(key + '  24\n')

    # tree_width = []
    # count = 0
    # with open('tree_width.txt', 'r') as fp:
    #     width_msg = fp.read().split('\n')
    #     # print(width_msg)
    #     for line in width_msg:
    #         fname = line.split('  ')[0]
    #         width = line.split('  ')[-1]
    #         # if short > depth:
    #         #     short = depth
    #         # if long < depth:
    #         #     long =  depth
    #         if int(width) > 93:
    #             count += 1
    #         tree_width.append(int(width))

    # tree_width = dict(Counter(tree_width))
    # tree_width = dict(sorted(tree_width.items(), key=operator.itemgetter(0)))

    # # print(tree_width)
    # print(len(tree_width))
    # print(count)

    # code_len = {}
    # code_dep = {}
    # with open('all_code_length.txt', 'r') as fp:
    #     fdata = fp.read().split('\n')
    #     for line in  fdata:
    #         fname = line.split('  ')[0]
    #         flen = line.split('  ')[-1]
    #         code_len[fname] = flen

    # with open('all_tree_depths.txt', 'r') as fp:
    #     fdata = fp.read().split('\n')
    #     # print(fdata)
    #     for line in  fdata:
    #         fname = line.split('  ')[0]
    #         fdep = line.split('  ')[-1]
    #         code_dep[fname] = fdep

    # # print(code_len)
    # with open('all_code_msg.txt', 'w') as fp:
    #     for fname, fdep in code_dep.items():
    #         fp.write(fname + '  ' + code_dep[fname] + '  ' + fdep + '\n')