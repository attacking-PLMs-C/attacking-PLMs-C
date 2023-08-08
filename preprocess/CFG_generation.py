import os
import glob
from multiprocessing import Pool
from functools import partial


def add_class(newdict, filepath):
    filename = filepath.split('/')[-1]
    with open(newdict + filename, "w", encoding="utf-8") as f1:
        with open(filepath, "r+", encoding="utf-8") as f:
            data = f.readlines()
        if data[0].split(' ')[0] == 'class' or data[0].split(' ')[1] == 'class':
            newdata = data
        else:
            methodname = data[0].split('(')[0].split(' ')[-1]
            newdata = ['public class ' + methodname + '{\n']
            newdata.extend(data)
            newdata.append('}')
            # print(newdata)
        for line in newdata:
            f1.writelines(line)


def joern_parse(file, outdir):
    name = file.split('/')[-1].split('.java')[0]
    out = outdir + name + '.bin'
    os.environ['file'] = str(file)
    os.environ['out'] = str(out)
    #print(file,out)
    os.system('joern-parse $file --output $out')  # --language c
    #print('bin ok')


def joern_export(bin, outdir):
    name = bin.split('/')[-1].split('.bin')[0]
    #print(name)
    out = outdir + name
    # if os.path.exists(out):
    #     pass
    # else:
    #     os.mkdir(out)
    os.environ['bin'] = str(bin)
    os.environ['out'] = str(out)
    #print(bin,out)
    os.system('joern-export $bin --repr cfg --out $out')
    #print('cfg ok')


def get_cfg(inputfile, output_path1, output_path2, type):

    if output_path1[-1] == '/':
        output_path1 = output_path1
    else:
        output_path1 += '/'

    if output_path2[-1] == '/':
        output_path2 = output_path2
    else:
        output_path2 += '/'
    # print(output_path1)
    if os.path.exists(output_path1):
        pass
    else:
        os.mkdir(output_path1)

    if os.path.exists(output_path2):
        pass
    else:
        # os.mkdir(output_path2)
        pass
    
    if type == 'parse':
        joern_parse(inputfile, output_path1)
    elif type == 'export':
        name = inputfile.split('/')[-1].split('.java')[0]
        binfile = output_path1 + name + '.bin'
        joern_export(binfile, output_path2)
    else:
        print('Type error!')


def changepath_and_addquotation(path, newpath):
    # filenamelist = os.listdir(path)
    #filenamelist = ['1760180']
    #a = 0
    # print(filenamelist)
    
    old_name = path + '0-cfg.dot'
    new_name = newpath +  path.split('-')[0].split('/')[-1] + '.dot'
    
    print(old_name + '->' + new_name)
    #a += 1
    with open(new_name, "w", encoding="utf-8") as f1:
        with open(old_name, "r+", encoding="utf-8") as f:
            data = f.readlines()
        for line in data:
            if "label" in line:
                ls = line.split('label = ')
                line = ''
                line += ls[0]
                line += 'label = '
                line += '\"'
                k = ls[1].split(',')[1]
                if len(ls[1].split(',')) > 2:
                    i = 1
                    k = ''
                    while i < len(ls[1].split(',')):
                        k += ls[1].split(',')[i]
                        k += ','
                        i += 1

                if "<SUB>" in k:
                    line += k.split(')<SUB>')[0]
                else:
                    line += k.split(')>')[0]
                line += '\" ]\n'
                #print(line)
            f1.writelines(line)
    #print(a)

def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "/" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data):  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)
    os.rmdir(path_data)

def cfg_generation(javapath, dotdict):
    #print(javapath, dotdict)
    newdict =  './'+ javapath.split('/')[-1].split('.')[0]+ 'temp/'
    #print(newdict)
    if os.path.exists(newdict):
        pass
    else:
        os.mkdir(newdict)

    add_class(newdict, javapath)
    #print(os.listdir(newdict))
    bindict = './'+ javapath.split('/')[-1].split('.')[0]+'joern-bin/'
    cfg_tempt =  './'+ javapath.split('/')[-1].split('.')[0]+'-temp-cfg/'

    get_cfg(newdict , bindict, cfg_tempt, 'parse')   # 对newdict里面的文件进行parse，保存在bindict里面
    get_cfg(newdict , bindict, cfg_tempt, 'export')  # 对bindict里面的文件进行export，生成的cfg文件夹们保存在cfg-tempt里面
    changepath_and_addquotation(cfg_tempt, dotdict)  # 对cfg-tempt里面cfg提取0-cfg并且删除label中多余的部分，放入dotdict
    # print('change ok')
    cfgpath = dotdict + javapath.split('/')[-1].split('.java')[0] + '.dot'
    # 清空newdict，bindict，cfg_tempt文件夹，否则下次使用joern时会重复生成之前的文件
    del_file(newdict)
    #print('clean newdict ok')
    del_file(bindict)
    #print('clean bin ok')
    del_file(cfg_tempt)
    #print('clean temp ok')
    return cfgpath


if __name__ == "__main__":
    cfg_generation("./pairs/8001867.java","./cfg-dot/")
    #changepath_and_addquotation(os.path.abspath(".") + '/cfg-joern/', os.path.abspath(".") + '/cfg-dot/')
