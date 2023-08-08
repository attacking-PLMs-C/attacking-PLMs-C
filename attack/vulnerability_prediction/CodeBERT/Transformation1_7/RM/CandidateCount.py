
import os 
import math 
import time
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('fid',type=str)
parser.add_argument('filepath',type=str, help='please input the count file path')
parser.add_argument('action',type=int,help='actions')
args=parser.parse_args()
 
def get_conut_result(file_dir):
    F=[]
    for root,dirs,files in os.walk(file_dir):
        for filename in files:
            if(os.path.splitext(filename)[1]=='.count'):  #os.path.splitext()分离文件名和扩展名
                F.append(filename)
    return F
 

def get_file_number(filename):
    count=0
    for i in filename:
        if(str.isnumeric(i)):
            count+=1
    number=filename[:count]
    return int(number)

def gen_random_data(files, fname):
    random.seed(time.time())
    for filename in files:
        absFile=os.path.join(filepath,filename)
        # print(filename)
        with open(absFile,'r') as fileHandle:
            filenumber=get_file_number(filename)
            count=int(fileHandle.readline().strip())
            if(action == filenumber):
                with open('all_candidates_count.txt', 'a') as fp:
                    fp.write(fname + '  ' + str(count) + '\n')
                with open('candidate_count.txt', 'w') as fp:
                    fp.write(str(count))

if __name__ == '__main__':
    fname = args.fid
    filepath = args.filepath    
    action = args.action
    files=get_conut_result(filepath)
    gen_random_data(files, fname)


