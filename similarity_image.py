import os
import pandas as pd
import copy
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import sys
import torch
import pandas as pd
import gc 
import numpy as np
import clip
import torchvision
import csv
import ast
import cv2


model, preprocess = clip.load("ViT-B/16")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def image_similarity(image, image_body, image_database, image_box_database,image_label_database, database_features, database_body_features,top_k=32):
 
    with torch.no_grad():
        query_image = preprocess(image).unsqueeze(0).to(device)
        query_body = preprocess(image_body).unsqueeze(0).to(device)
        image_features = model.encode_image(query_image).float()
        body_features = model.encode_image(query_body).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        body_features /= body_features.norm(dim=-1, keepdim=True)  
       
        similarities_image = torch.matmul(image_features.unsqueeze(0), database_features.transpose(1, 2))
        similarities_body = torch.matmul(body_features.unsqueeze(0), database_body_features.transpose(1, 2))
        similarities_image = similarities_image.squeeze(-1).squeeze(-1).cpu()
        similarities_body = similarities_body.squeeze(-1).squeeze(-1).cpu()
        
        similarities = (similarities_body + similarities_image)/2
        top_k_indices = similarities.argsort(descending=True)[:top_k]

        top_matches = [image_database[i] for i in top_k_indices]
        top_matches_box = [image_box_database[i] for i in top_k_indices]
        top_matches_label = [image_label_database[i] for i in top_k_indices]
    
    return top_matches,top_matches_box,top_matches_label

# for heco
query_images_path = "dataset/HECO/test.csv"
sampled_images = 'sample_base/heco_sampled_images.csv'
save_path = 'dataset/heco_testdata_with_examples.csv'

# for emotic
# query_images_path = "dataset/Emotic/test.csv"
# sampled_images = 'sample_base/emotic_sampled_images.csv'
# save_path = 'dataset/emotic_testdata_with_examples.csv'

df_query = pd.read_csv(query_images_path)
df_example = pd.read_csv(sampled_images)
examples = []
examples_box = []
examples_label = []
image_database = []
image_box_database = []
image_label_database = []
database_features = []
database_body_features = []

with torch.no_grad():
    for i in range(len(df_example)):
        
        # for heco
        image_path = 'dataset/HECO/Data/'+df_example['Image'][i]
        x_min, y_min, x_max, y_max = int(df_example['xmin'][i]), int(df_example['ymin'][i]), int(df_example['xmax'][i]), int(df_example['ymax'][i])
        label = df_example['Category'][i]
        
        # for emotic
        # image_path = 'dataset/Emotic/emotic/' + df_example['Folder'][i] +'/'+ df_example['Filename'][i]
        # bbox = ast.literal_eval(df_example['BBox'][i])
        # x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        # label = df_example['Categorical_Labels'][i]
        
        image = Image.open(image_path)
        image_body = image.crop((x_min, y_min, x_max, y_max))
        
        image_context = copy.deepcopy(cv2.imread(image_path))
        image_context[y_min:y_max, x_min:x_max] = (0, 0, 0)
        image = cv2.cvtColor(image_context, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        img = preprocess(image).unsqueeze(0).to(device)
        body = preprocess(image_body).unsqueeze(0).to(device)
        
        image_database.append(image_path)
        image_box_database.append([x_min, y_min, x_max, y_max])
        image_label_database.append(label)
        
        database_features.append(model.encode_image(img).float())
        database_body_features.append(model.encode_image(body).float())
        
database_features = torch.stack(database_features) / torch.stack(database_features).norm(dim=-1, keepdim=True)
database_body_features = torch.stack(database_body_features) / torch.stack(database_body_features).norm(dim=-1, keepdim=True)

for i in range(len(df_query)):
    
    # for emotic
    # image_path = 'dataset/Emotic/emotic/' + df_query['Folder'][i] +'/'+ df_query['Filename'][i]
    # bbox = ast.literal_eval(df_query['BBox'][i])
    # x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # for heco
    image_path = 'dataset/HECO/Data/'+df_query['Image'][i]
    x_min, y_min, x_max, y_max = int(df_query['xmin'][i]), int(df_query['ymin'][i]), int(df_query['xmax'][i]), int(df_query['ymax'][i])
    
    image = Image.open(image_path)
    image_body = image.crop((x_min, y_min, x_max, y_max))

    image_context = copy.deepcopy(cv2.imread(image_path))
    image_context[y_min:y_max, x_min:x_max] = (0, 0, 0)
    image = cv2.cvtColor(image_context, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    top_matches, top_matches_box, top_matches_label = image_similarity(image, image_body, image_database, image_box_database,image_label_database, database_features,database_body_features)

    examples.append(top_matches)
    examples_box.append(top_matches_box)
    examples_label.append(top_matches_label)
    
df_query['examples'] = examples
df_query['examples_box'] = examples_box
df_query['examples_label'] = examples_label
df_query.to_csv(save_path)
        