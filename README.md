# Attacking pre-trained language models of code
This repository provides the code and data of the submitted paper: "Attacking Pre-Trained Language Models of Code with Multi-level Linguistic Representations"

Our approach contains two parts: (1) probing tasks; (2) MindAC.

# Requirements
python                    3.8.13

numpy                     1.21.2

pandas                    1.3.4

torch                     2.0.0+cu118

tqdm                      4.63.0

scikit-learn              1.0.1

transformers              4.20.1

TXL                       v10.8 (7.5.20) 

# Datasets
We experiment on two open source C/C++ datasets: (1) Devign dataset; (2) POJ-104 dataset.
Devign dataset contains FFmpeg and QEMU datasets, which are available at https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view?pli=1.

POJ-104 dataset is available at https://drive.google.com/file/d/0B2i-vWnOu7MxVlJwQXN6eVNONUU/view?resourcekey=0-Po3kiAifLfCCYnanCBDMHw

# Research questions

In this paper, we set three research questions: 

(1) RQ1: Which layers of the victim models are significant for linguistic features learning?

(2) RQ2: How effective is the linguistic representations-based attack compared with the state-of-the-art baselines?

(3) RQ3: Can we improve the robustness of victim models with adversarial examples?

# Answer for RQ1: Probing tasks

Before the probing tasks, we need to fine-tune the CodeBERT model.

python codebert.py --train_eval train --layer 12

We perform the probing tasks (i.e., 2 surface probing tasks, 3 syntax probing tasks and 3 probing tasks) on Devign dataset.

cd ./probing

Surface probing tasks. Running the code in the "code_length" and "code_content" folders. For example, to perform "CodeLength" task, we need to run the following scripts:

cd ./code_length

python code_length.py  # labeling each code snippet according to its length

python tokenization.py  

python codebert.py --train_eval prob --layer 1  # evaluating the ability of the first layer

Before performing the probing tasks, we use the


# MindAC
