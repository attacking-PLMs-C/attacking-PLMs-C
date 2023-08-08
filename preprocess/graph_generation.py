import os
import json
import tqdm
import argparse
import subprocess
from multiprocessing import Pool

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--g",
                        type=str,
                        choices=['ast', 'cfg'],
                        help="input graph type",
                        )
    return parser.parse_args()

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

def generate_tmp(dir_name):
    root = './valid_code/' + dir_name.split('_')[-1]
    tmp = './tmp/' + dir_name  + '.bin'
    sh = "joern-parse " + root + " --output " + tmp
    try:
        subprocess.check_output(sh, shell=True)
    except Exception as e:
        pass

def generate_graph(dir_name):
    g_dir = './cfg_dot/' + dir_name
    # g_svg = './ast_svg/' + dir_name + '-' + str(args.g) + '.svg'
    g_dot = g_dir  + '/1-' + args.g + '.dot'
    tmp = './tmp/' + dir_name  + '.bin'
    os.environ['gtype'] =  args.g
    # os.environ['root'] = root
    os.environ['tmp'] = tmp
    os.environ['g_dir'] = g_dir
    os.environ['g_dot'] = g_dot
    # os.environ['g_svg'] = g_svg
    # os.system('joern-parse $root --output $tmp')
    try:
        os.system('joern-export $tmp --repr $gtype --out $g_dir')
    except Exception as e:
        pass
    # os.system('dot -Tsvg $g_dot -o $g_svg')

args = get_parameter()

if __name__ == "__main__":
    file_path_list = []
    get_file_path("./valid_code", file_path_list)
    arg1_list = []
    for path in file_path_list:
        dir_name = path.split('/')[-1].split('.')[0]
        # root = './valid_code/' + dir_name.split('_')[-1]
        # tmp = './tmp/' + dir_name  + '.bin'
        # g_dir = './ast_dot/' + dir_name
        # g_svg = './ast_svg/' + dir_name + '-' + str(args.g) + '.svg'
        # arg1_list.append([args.g, root, tmp, g_dir, g_svg])
        arg1_list.append(dir_name)

    '''生成bin文件'''
    # bar = tqdm.tqdm(arg1_list)
    # p = Pool(2)
    # for i in p.imap(generate_tmp, arg1_list):
    #     bar.update()
    # bar.close()

    '''生成ast和cfg的dot和svg文件'''
    bar = tqdm.tqdm(arg1_list)
    p = Pool(2)
    for i in p.imap(generate_graph, arg1_list):
        bar.update()
    bar.close()
    # print(arg1_list[0])
    # print(arg1_list[1])
    # for item in tqdm.tqdm(arg1_list):
    #     generate_graph(item[0], item[1], item[2], item[3], item[4])
    #     break
    # with Pool(40) as p:
    #     for i in p.imap(generate_graph, path_list):
            
