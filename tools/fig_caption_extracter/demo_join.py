import numpy as np
import json
import pdb

import os

def calcul_value(annotation_list):
    # 统计所有value的次数
    # Count the number of all values
    all_vlaue = {}
    for i in annotation_list:
        k = i['image_id']
        v = i['caption']
        if v in all_vlaue:
            all_vlaue[v] = all_vlaue.get(v) + 1
        else:
            all_vlaue[v] = 1
    return all_vlaue

def get_repeat_value(value_cal):
    # 删除出现一次的value
    # Delete values that appear once
    repeat_vlaue = {}
    for k, v in value_cal.items():
        if v != 1:
            repeat_vlaue[k] = v
    return repeat_vlaue

def del_repeat_value(annotations_list, value_cal,repeat_vlaue):
    # 删除重复的value，只保留第一次出现的
    # Delete the redundant duplicate values
    annotations_result = []
    for i in annotations_list:
        k = i['image_id']
        v = i['caption']
        if (value_cal.get(v) == 1):
            annotations_result.append({'image_id':k, 'caption':v})
        if ((value_cal.get(v) >= 2) & (value_cal.get(v) == repeat_vlaue.get(v))):
            annotations_result.append({'image_id':k, 'caption':v})
            value_cal[v] = value_cal.get(v) + 1
    return annotations_result


pj_path = r"PATH TO THE PDF FOLDER/PROJECT"
annotations = []
for root,dirs,files in os.walk(pj_path):
    for fi in files:
        if fi.endswith('.json'):
            print(os.path.join(root,fi))
            with open(os.path.join(root,fi),'r',encoding='utf-8') as jf:
                records = json.load(jf)
                annotations_list = records['anotations']
                value_cal = calcul_value(annotations_list)
                repeat_value = get_repeat_value(value_cal)
                remove_repeat_records = del_repeat_value(annotations_list,value_cal,repeat_value)

                real_path = os.path.relpath(root,pj_path)
                for ai in remove_repeat_records:
                    img_path = os.path.join('./',real_path,"media_convert",ai['image_id']+".jpeg")
                    if os.path.exists(img_path):
                        annotations.append({"image_id": img_path, "caption": ai['caption']})

print(len(annotations))
json_out = {'anotations': annotations}
with open("test.json", 'w', encoding='utf-8') as jf:
    jf.write(json.dumps(json_out, ensure_ascii=False, indent=3))


