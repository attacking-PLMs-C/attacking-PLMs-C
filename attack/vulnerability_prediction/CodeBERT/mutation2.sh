#!/bin/bash

# 记录开始时间
start_time=$(date +"%Y-%m-%d %H:%M:%S")

# 串行运行Transformation1_12文件夹下的code_trans.py文件
cd Transformation1_12 && python code_trans.py

cd ..

# 并行运行Transformation1_12_2文件夹下的code_trans.py文件
(cd Transformation1_12_2 && python code_trans.py) &

# 并行运行Transformation1_12_3文件夹下的code_trans.py文件
(cd Transformation1_12_3 && python code_trans.py) &

# 并行运行Transformation1_12_7文件夹下的code_trans.py文件
(cd Transformation1_12_7 && python code_trans.py) &

# 等待所有后台任务完成
wait

# 记录结束时间
end_time=$(date +"%Y-%m-%d %H:%M:%S")

# 计算运行时间
start_seconds=$(date --date="$start_time" +%s)
end_seconds=$(date --date="$end_time" +%s)
elapsed_seconds=$((end_seconds - start_seconds))

# 打印运行时间
echo "总运行时间：$elapsed_seconds 秒"
