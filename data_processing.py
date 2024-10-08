import json
from tqdm import tqdm
import re
import json
import random
from copy import deepcopy
import string
from collections import OrderedDict
random.seed(12)

def convert_to_xlam_tool(tools):
    ''''''
    if isinstance(tools, dict):
        return {
            "name": tools["name"],
            "description": tools["description"],
            "parameters": {k: v for k, v in tools["parameters"].get("properties", {}).items()}
        }
    elif isinstance(tools, list):
        return [convert_to_xlam_tool(tool) for tool in tools]
    else:
        return tools

def build_prompt(task_instruction: str, format_instruction: str, tools: list, query: str):
    prompt = f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{tools}\n[END OF AVAILABLE TOOLS]\n\n"#json.dumps(tools)
    prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    return prompt


task_instruction_hammer = """You are a tool calling assistant. In order to complete the user's request, you need to select one or more appropriate tools from the following tools and fill in the correct values for the tool parameters. Your specific tasks are:
1. Make one or more function/tool calls to meet the request based on the question.
2. If none of the function can be used, point it out and refuse to answer.
3. If the given question lacks the parameters required by the function, also point it out.
"""
format_instruction_hammer = """
The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please directly output an empty list '[]'
```
[
    {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
    ... (more tool calls as required)
]
```
"""

def get_hammer_prompt(it):
    openai_format_tools = it['tools']
    query = it['query']
    label = it['answers']#convert_parameter_format(it['parameters'],it['interface'],if_pandding)
    xlam_format_tools = convert_to_xlam_tool(openai_format_tools)
    label = f"""```
{label}
```"""
   
    content = build_prompt(task_instruction_hammer, format_instruction_hammer, xlam_format_tools, query)
    return {
        "instruction": content,
        "input": "",
        "output": label
    }



def replace_param_names_new(data):
    
    #letters = string.ascii_lowercase
    letters = list(string.ascii_uppercase)+list(string.ascii_lowercase)+['_','.']+list(map(str,range(10)))
    new_data = deepcopy(data)
    tools = []
    t_name= []
    old_tools = json.loads(new_data['tools'])
    N =len(old_tools)
    for t in old_tools:
        if t['name'] not in t_name:
            tools.append(t)
            t_name.append(t['name'])
    answers = json.loads(data['answers'])
    
    
    i = 0
    for tool in tools:
        
        keys = list(tool['parameters'].keys())
        for param in keys:
            old_name = param
            new_name = ''.join(random.choice(letters) for i in range(random.randint(4,10)))  # 生成随机字符串
            tool['parameters'][new_name] = tool['parameters'].pop(old_name)
            if len(new_data['answers']):
                for answer in answers:
                    if old_name in answer['arguments'] and answer['name']==tool["name"]:
                        answer['arguments'][new_name] = answer['arguments'].pop(old_name)
    if len(tools)!=N:
        tools=tools+random.choices(tools,k=random.randint(0,N-len(tools)+1))
    random.shuffle(tools)
    new_data['tools'] = json.dumps(tools)
    new_data['answers'] = json.dumps(answers)

    return new_data


def replace_function_names_new(data):
    letters = list(string.ascii_uppercase)+list(string.ascii_lowercase)+['_','.']+list(map(str,range(10)))
    answers = json.loads(data['answers'])
    
    new_data = deepcopy(data)
    tools = []
    t_name= []
    old_tools = json.loads(new_data['tools'])
    N =len(old_tools)
    for t in old_tools:
        if t['name'] not in t_name:
            tools.append(t)
            t_name.append(t['name'])
    i = 0
    for tool in tools:
        old_name = tool['name']
        new_name = new_name = ''.join(random.choices(letters,k=random.randint(5,15))) 
        tool['name'] = new_name
        if len(new_data['answers']):
            for answer in answers:
                if answer['name'] == old_name:
                    answer['name'] = new_name
        
        
        i+=1
    if len(tools)!=N:
        tools=tools+random.choices(tools,k=random.randint(0,N-len(tools)+1))
    random.shuffle(tools)
    new_data['tools'] = json.dumps(tools)
    new_data['answers'] = json.dumps(answers)
    
    return new_data


def generate_random_value(defalt):
    if type(defalt)!=str:
        if type(defalt) == int:
            defalt+=random.randint(1,4)
        else:
            defalt+=random.random()
        return defalt
    letters = list(string.ascii_uppercase) + list(string.ascii_lowercase) + ['_', '.'] + list(map(str, range(10)))
    return ''.join(random.choice(letters) for _ in range(5))
def replace_in_query(query, old_value, new_value):
    # Replace all occurrences of old_value with new_value in query, case insensitive
    old_value, new_value = str(old_value), str(new_value)
    query = query.replace(old_value, new_value)
    query = query.replace(old_value.capitalize(), new_value.capitalize())
    query = query.replace(old_value.lower(), new_value.lower())
    query = query.replace(old_value.upper(), new_value.upper())
    return query
def replace_in_des(des, old_value, new_value):
    # Replace all occurrences of old_value with new_value in query, case insensitive
    des = des.replace(str(old_value), str(new_value))

    return des

def replace_param_default_values_news(data):
    new_data = deepcopy(data)
    #tools = json.loads(new_data['tools'])
    tools = []
    t_name= []
    old_tools = json.loads(new_data['tools'])
    N =len(old_tools)
    for t in old_tools:
        if t['name'] not in t_name:
            tools.append(t)
            t_name.append(t['name'])
    answers = json.loads(new_data['answers'])

    

    for tool in tools:
        
        for param, param_info in tool['parameters'].items():
            default_value = param_info.get('default', None)
            
            if type(default_value) == list:
                continue
            if default_value==None or default_value=='':
                continue
            
            keep = 0
            for ans in answers:
                if ans['name'] != tool['name']:
                    continue
                
                if param in ans["arguments"] and str(default_value)==str(ans["arguments"][param]):
                    keep=1

                    break
                
            if keep==0:
                continue
                        
            
            
            # Randomly generate a new default value
            new_default_value = generate_random_value(deepcopy(default_value))
            
            # Replace in tool's parameters
            tool['parameters'][param]['default'] = new_default_value
            tool['parameters'][param]['description'] = replace_in_des(tool['parameters'][param]['description'], default_value, new_default_value)
            # Replace in answers
            for answer in answers:
                if answer['name'] == tool['name'] and param in answer['arguments']:
                    argument_value = answer['arguments'][param]
                    if default_value==argument_value:
                        answer['arguments'][param] = new_default_value
                        
                        new_data['query'] = replace_in_query(new_data['query'], default_value, new_default_value)
    
    new_data['tools'] = json.dumps(tools)
    new_data['answers'] = json.dumps(answers)
    
    return new_data

def check_default_values_news(data):
    new_data = deepcopy(data)
    #tools = json.loads(new_data['tools'])
    tools = []
    t_name= []
    old_tools = json.loads(new_data['tools'])
    N =len(old_tools)
    for t in old_tools:
        if t['name'] not in t_name:
            tools.append(t)
            t_name.append(t['name'])
    answers = json.loads(new_data['answers'])

    

    for tool in tools:
        
        for param, param_info in tool['parameters'].items():
            default_value = param_info.get('default', None)
            
            if type(default_value) == list:
                continue
            if default_value==None or default_value=='':
                continue
            
            keep = 0
            for ans in answers:
                if ans['name'] != tool['name']:
                    continue
                
                if param in ans["arguments"] and str(default_value)==str(ans["arguments"][param]) and str(default_value).lower() not in data['query'].lower():
                    keep=1

                    break
                
            if keep==0:
                continue
            else:
                return True
    return False



# get original data
data= json.load(open('data/xlam_function_calling_60k.json'))
reject = json.load(open('data/XLAM-7.5k-Irrelevance.json','r'))
# copy 3x and then perform masking processing
data_three = []
random.seed(12)
data_one = data+reject
random.shuffle(data_one)
data_three+=data_one
data_one = data+reject
random.shuffle(data_one)
data_three+=data_one
data_one = data+reject
random.shuffle(data_one)
data_three+=data_one

# function making
random.seed(12)
data_mask = []
for it in tqdm(data_three):
    if random.random()>1/3:#Masking ratio p=1-1/3=0.67
        data_mask.append(replace_param_default_values_news(replace_param_names_new(replace_function_names_new(it)))) 
    else:       
        data_mask.append(it)
    

# get sft data
sft_mask_hammer = [get_hammer_prompt(it) for it in data_mask]
with open('data/masking_sft_data.json','w') as f:
    json.dump(sft_mask_hammer,f,ensure_ascii=False,indent=2)

                