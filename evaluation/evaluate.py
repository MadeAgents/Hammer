import json
import sys
import os
import pandas as pd
import re
from tqdm import tqdm

def parse_toolname_and_parameters_hammer(input_text):
    if '```json' in input_text:
        input_text = input_text.replace("```json", '')
    input_text = input_text.replace("```", '')
    dic = json.loads(input_text)
    if 'tool_calls' in dic:
        dic = dic['tool_calls']
    if dic==None:
        dic=[]
    return dic

def get_p_r_exact(param_dict0, param_dict):
    num0, num, acc = len(param_dict0), len(param_dict), 0
    if num0==0 and param_dict0==param_dict:
        return 1, 1, 1
    if type(param_dict) != dict:
        return num0, 0, 0
    param_dict0 = {k: str(v) for k, v in param_dict0.items() if v != ''}
    param_dict = {k: str(v) for k, v in param_dict.items() if v != ''}
    if num0==0 and param_dict0==param_dict:
        return 1, 1, 1
    for k, v in param_dict0.items():
        if k not in param_dict.keys():
            score = 0
        else:
            score = v== param_dict[k]
        if score:
            acc += 1
    return num0, num, acc



if '.jsonl' in sys.argv[1]:
    output_file = sys.argv[1]
else:
    output_file = sys.argv[1]+"/generated_predictions.jsonl"  

with open(output_file, 'r') as infile:
    outputs = [json.loads(line) for line in infile]

name_acc = 0
tool_rej = 0
json_format = 0
param0_num, param_num, param_correct = 0, 0, 0
results = []
correct_api_num = 0
predict_api_num = 0
gold_api_num = 0

correct_param_num = 0
predict_param_num = 0
gold_param_num = 0
for x in outputs:

    
    label = parse_toolname_and_parameters_hammer(x['label'])
        
    gold_api_num += len(label)
    gold_api_name = []
    for gold_api in label:
        
        gold_param_num += len([k for k,v in gold_api['arguments'].items() if v!=''])
        gold_api_name.append(gold_api['name'])
        
    try:
        predict = parse_toolname_and_parameters_hammer(x['predict'])
    except:
        predict = '<format-err>'
    if type(predict)!=list:
        predict_api_num+=1
        predict = '<format-err>'
    if predict=='<format-err>' or len(predict)==0:
        continue
    
    predict_api_num+=len(predict)
    
    for predict_api in predict:
        
        predict_param_num+=len([k for k,v in predict_api['arguments'].items() if v!=''])
        

        try:
        
            if predict_api['name'] in gold_api_name:
                correct_api_num += 1
                correct0 = 0
                for idx in range(len(label)):
                    if label[idx]['name']==predict_api['name']:
                        num0, num, correct = get_p_r_exact(label[idx]['arguments'], predict_api['arguments'])
                        if correct>correct0:
                            correct0 = correct
                            break
                correct_param_num+=correct0
        except:
            continue
        

result_dict = {}
if correct_api_num * predict_api_num * gold_api_num > 0:
    result_dict["P_api"] = 1.0*correct_api_num/predict_api_num
    result_dict["R_api"] = 1.0*correct_api_num/gold_api_num
    result_dict["F1_api"] = 2*result_dict["P_api"]*result_dict["R_api"]/(result_dict["P_api"]+result_dict["R_api"])

if correct_param_num * predict_param_num * gold_param_num > 0:
    result_dict["P_param"] = 1.0*correct_param_num/predict_param_num
    result_dict["R_param"] = 1.0*correct_param_num/gold_param_num
    result_dict["F1_param"] = 2*result_dict["P_param"]*result_dict["R_param"]/(result_dict["P_param"]+result_dict["R_param"])

print(result_dict)
       

