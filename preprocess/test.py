import os
import tqdm
import re

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
    # file_path_list = []
    # get_file_path('./valid_code', file_path_list)
    # bar = tqdm.tqdm(file_path_list)
    # for path in bar:
    #     fname = path.split('/')[-1].split('.')[0]
    #     # print(fname)
    #     dot_path =  './ast_dot/' + fname + '/1-ast.dot'
    #     fdata = open(dot_path, 'r').read().split('\n')
    #     # print(fdata[1])
    #     # print(fdata[3])
    #     all_nodes ={}
    #     cur_depth = {}     #将每一个子节点深度设为1
    #     tag = True         #root节点标签
    #     root = ''
    #     tree_depth = 0
    #     for line in fdata:
    #         # print(line)
    #         # num = line.split('" [')[0].split('"')[-1]
    #         # pattern = re.compile(r'block_crypto_open_opts_init')
    #         # node_data = re.search(r'<.*</SUB>>', line)
    #         # if node_data != None and num != None:
    #         #    node_data = node_data.group()
    #         #    print(num)
    #         #    print(node_data)
    #         # print('-------------------------------')
    #         edge_data = re.search(r'.*" -> ".*', line)
    #         if edge_data != None:
    #             edge_data  = edge_data.group().strip()
    #             start = edge_data.split('"')[1]
    #             end = edge_data.split('"')[-2]
    #             # print(start + '  ' + end)
    #             if tag == True:
    #                 root = start
    #                 tag = False
    #             if start not in all_nodes:
    #                 all_nodes[start] = []
    #             all_nodes[start].append(end)
    #             if start not in cur_depth:
    #                 cur_depth[start] =  1
    #             if end not in cur_depth:
    #                 cur_depth[end] = 1
    #         # break
    #     print(all_nodes)
    #     print(root)
    #     print(cur_depth)
    #     for node in all_nodes:
    #         for sub_node in all_nodes[node]:
    #             cur_depth[sub_node] += cur_depth[node]
    #             if tree_depth < cur_depth[sub_node]:
    #                 tree_depth  = cur_depth[sub_node]   #每个子节点深度等于“1+父节点所处深度”

    #     print(tree_depth)

    #     break

    # # get_file_path("./valid_code_0", file_path_list)
    # # bar = tqdm.tqdm(file_path_list)
    # # for path in bar:
    # #     fname = path.split('/')[-1].split('.')[0]
    # #     new_dir = fname.split('_')[-1]
    # #     source_file = './valid_code_0/' + path.split('/')[-1]
    # #     aim_file = './valid_code/' + new_dir
    # #     if not os.path.exists(aim_file):
    # #         os.mkdir(aim_file)
    # #     # print(aim_file)
    # #     # break
    # #     os.environ['source_file'] = source_file
    # #     os.environ['aim_file'] = aim_file
    # #     os.system('cp $source_file $aim_file')
    # #     bar.update()
    # # bar.close()

    tree_depth = 0
    all_nodes ={'7': ['8']}
    cur_depth = {'7': 1, '8': 1} 
    for node in all_nodes:
            for sub_node in all_nodes[node]:
                cur_depth[sub_node] += cur_depth[node]
                if tree_depth < cur_depth[sub_node]:
                    tree_depth  = cur_depth[sub_node]   #每个子节点深度等于“1+父节点所处深度”

    print(tree_depth)