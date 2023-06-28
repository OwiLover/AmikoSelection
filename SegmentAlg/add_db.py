import sqlite3
import os
import numpy as np
import torch
import clip
from PIL import Image

def addPerson(id, name):
    con = sqlite3.connect('test_info.db', check_same_thread=False)
    cur=con.cursor()
    cur.execute("INSERT INTO Users (Vector,Destination) VALUES (?,?)",(str(id) , str(name))),
    con.commit()
    con.close()


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device)

img = Image.open('./cut_images/person1164.jpg').convert('RGB')

image_input = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)

input_path = './People_2 26'

files = os.listdir(input_path)


print(len(image_features[0]))

vec_mat = np.zeros((len(files), len(image_features[0])))

sample_indices = np.random.choice(range(0, len(files)), size=len(files), replace=False)

destination = "Beauty Technique"
print('Reading images...')
for index, i in enumerate(sample_indices):
    file = files[i]
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(input_path, filename)).convert('RGB')
    image_input = preprocess(img).unsqueeze(0).to(device)
    vec = model.encode_image(image_input)

    vector = vec[0].T.cpu().detach().numpy()
    addPerson(vector, destination)

print(vec_mat)