# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
import torch
from PIL import Image

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
import csv
import pandas as pd
import ast
import cv2
import copy
from tqdm import tqdm

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_path,box):
    image = copy.deepcopy(cv2.imread(image_path))
    x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def load_images(image_files, boxs):
    out = []
    for image_file, box in zip(image_files, boxs):
        image = load_image(image_file, box)
        out.append(image)
    return out

def eval_model(args):
    disable_torch_init()
        
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, model_name, args.model_base
    )
    
    df = pd.read_csv(args.path_annotations)
    df_examples = pd.read_csv('sample_base/emotic_sampled_images_analysis.csv')
    pred_s = []
    path_result = os.path.join(args.path_result,'emotic_result.csv')
    with open(path_result, 'w') as csvfile:
        
        filewriter = csv.writer(csvfile, delimiter=',', dialect='excel')
        row = ['Folder', 'Filename','Image Size','BBox','Categorical_Labels','examples','preds']
        filewriter.writerow(row)
        total_samples = len(df)
        for i, row in tqdm(df.iterrows(), total=total_samples, desc="Inference Progress"):   
            image_path = 'dataset/Emotic/emotic/'+df['Folder'][i]+'/'+df['Filename'][i]
            image_box = ast.literal_eval(df['BBox'][i])
            
            image_paths = []
            image_boxs = []
            
            true_label = df['Categorical_Labels'][i]
            qs = args.query
            
            if args.n_shot > 0:
                examples = ast.literal_eval(df['examples'][i])
                examples = examples[:args.n_shot]
                
                for j in range(args.n_shot):
                    labels_j = ast.literal_eval(df['examples_label'][i])[j]
                    box = ast.literal_eval(df['examples_box'][i])[j]
                    
                    analyzation_indexs = df_examples[df_examples.Path == examples[j].split('emotic/')[-1]].index.tolist()  
                    analyzation_index = analyzation_indexs[0] 
                    for k in analyzation_indexs:
                       
                        if box == ast.literal_eval(df_examples['BBox'][k]):
                            analyzation_index = k
                          
                    
                    analyzation = df_examples['Analysis'][analyzation_index]
                    lines = analyzation.splitlines()
                    non_empty_lines = [line for line in lines if line.strip() != ""]
                    analyzation = "\n".join(non_empty_lines)
                
                    image_paths.append(examples[j])
                    image_boxs.append(box)
                    
                    qs += f' Example {j+1}: Image: {DEFAULT_IMAGE_TOKEN}\n Answer: {analyzation}\n The final answer is {labels_j}\n'
                qs += f'Question: Image: {DEFAULT_IMAGE_TOKEN}\n Answer:'
                
                
            image_paths.append(image_path)
            image_boxs.append(image_box)
            images = load_images(image_paths,image_boxs)
            
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if DEFAULT_IMAGE_TOKEN not in qs:
                    # do not repeatively append the prompt.
                    if model.config.mm_use_im_start_end:
                        qs = (image_token_se + "\n") * len(images) + qs
                    else:
                        qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
      
            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            else:
                conv_mode = "llava_v0"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                pass
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
 
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[
                        images_tensor,
                    ],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

           
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
          
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            
            row = [df['Folder'][i], df['Filename'][i], df['Image Size'][i], df['BBox'][i], true_label, df['examples'][i], outputs]
            filewriter.writerow(row)

if __name__ == "__main__":
    
    path_annotations = 'dataset/emotic_testdata_with_examples.csv'
    output_dir = 'results/emotic'
    #for emotic
    class_names =  ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
            'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness', \
            'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    # for heco
    # class_names =['Surprise', 'Excitement', 'Happiness', 'Peace', 'Disgust', 'Anger', 'Fear', 'Sadness']
    
    query =  f"""Given the list of emotion labels: {str(class_names)}, please analyse and choose which emotions are more suitable for describing how the person in the red box feels.\n"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weight_base/Llama-3-VILA1.5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--path_annotations", type=str, default=path_annotations)
    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--path_result", type=str, default=output_dir)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, default=query)
    parser.add_argument("--n_shot", type=int, default=6)
    parser.add_argument("--conv-mode", type=str, default='llama_3')
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)