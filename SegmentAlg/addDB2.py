import torch
import numpy as np
from torch import nn
import os
import cv2 as cv
import clip
import sqlite3
from sklearn.cluster import KMeans
import pandas as pd
import joblib
from PIL import Image
import select_db as db
import time
from sklearn.model_selection import train_test_split
import joblib
import psutil


con = sqlite3.connect('test_info.db', check_same_thread=True)

def addPerson(id, name, con):
    cur=con.cursor()
    cur.execute("INSERT INTO Users (Vector,Destination) VALUES (?,?)",(str(id) , str(name))),
    con.commit()
    con.close()

def addBD():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load('ViT-L/14@336px', device)

    cap = cv.VideoCapture("people.mp4")

    name = "Enter camera location"
    while True:
        
        flag, img = cap.read()
        if flag:

            results = model(img)

            pred = results.pandas().xyxy[0]
            dfm = pd.DataFrame(pred)
            info = dfm.to_numpy()
            if info.size > 0:
                for item in info:
                    if item[6] == 'person':
                        xmin,ymin,xmax,ymax,chance = int(item[0]),int(item[1]),int(item[2]),int(item[3]), item[4]
                        if (xmin<xmax and ymin<ymax and chance > 0.9):
                            crop = img[ymin:ymax,xmin:xmax]
                            # cv.imshow('output', crop)
                            out = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
                            pil = Image.fromarray(out).convert('RGB')
                            image_input = preprocess(pil).unsqueeze(0).to(device)
                            with torch.no_grad():
                                image_features = model2.encode_image(image_input)
                                out = image_features[0].T.cpu().detach().numpy()
                            addPerson(out, name, con)

        ch = cv.waitKey(5)
        if ch == 27:
            break
    cap.release()
    cv.destroyAllWindows()

# addBD()