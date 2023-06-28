import torch
import numpy as np
from torch import nn
import os
import cv2 as cv
import clip
import sqlite3
import pandas as pd
import joblib
from PIL import Image
import select_db as db
import time
from sklearn.model_selection import train_test_split
# import psutil

con = sqlite3.connect('test_info.db', check_same_thread=True)
con2 = sqlite3.connect('.\AmikoBot\Amiko.db', check_same_thread=True)

def get_classes():
    array = []
    with open('classes.txt') as fp:
        for line in fp:
            x = line[:-1]
            array.append(x)
    return array

def createNewMlp():
    
    MLP = nn.Sequential(
        nn.Linear(768, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
        nn.Sigmoid()
    )

    vec, place = db.showUrOff(con)

    training_vec, validation_vec = train_test_split(vec, test_size = 0.8, random_state = 21)
    training_place, validation_place = train_test_split(place, test_size = 0.8, random_state = 21)

    training_places, nums = np.unique(training_place, return_inverse=True)
    print(training_places)

    with open('classes.txt', 'w') as fp:
        for item in training_places:
            fp.write("%s\n" % item)

    # print(nums[23])
    train_arrays= []
    for item in nums:
        i=0
        train_array = []
        while i<5:
            if i == item:
                train_array.append(1)
            else: train_array.append(0)
            i+=1
        train_arrays.append(train_array)
    # print(training_place)
    # print(train_arrays)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(MLP.parameters(), lr=1e-4)

    for epoch in range(0, 30):
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0
        for i, data in enumerate(training_vec,0):
            inputs = torch.FloatTensor(training_vec[i])
            targets = torch.FloatTensor(train_arrays[i])
            optimizer.zero_grad()
            outputs = MLP(inputs)
            # print(outputs)
            # print(targets)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
                current_loss = 0.0
    

    # Process is complete.
    print('Training process has finished.')

    val_places, nums = np.unique(validation_place, return_inverse=True)
# print(training_places)
# print(nums[1])


    def get_num_class(nums):
        train_arrays= []
        for item in nums:
            i=0
            train_array = []
            while i<5:
                if i == item:
                    train_array.append(1)
                else: train_array.append(0)
                i+=1
            train_arrays.append(train_array)
        return train_arrays

    ress = get_num_class(nums)

    val_vec = torch.FloatTensor(validation_vec)
    classes = get_classes()
    arr1 = []

    x=0
    for i in val_vec:
        # start = time.time()
        res= MLP(i)
        # end = time.time() - start
        # print(end)
        # print("Predicted: ")
        z = 0
        pred = ""
        for h in ress[x]:
            if h == 1:
                pred = classes[z]
                # print(classes[z])
            z+=1
        # print()
        # print(" Got: ")
        list = res.tolist()
        max_index = list.index(max(res))
        # print(classes[max_index])
        # print()
        # print()
        if (pred == classes[max_index]):
            arr1.append(1)
        else: arr1.append(0)
        x+=1

    summ = 0
    for item in arr1:
        summ +=item

    coef = summ/len(arr1)
    print("Схожесть: " + str(coef) +"%")
    joblib.dump(MLP, 'MLP2.joblib')



def maincam():

    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load('ViT-L/14@336px', device)
    model3 = joblib.load("MLP.joblib")
    cap = cv.VideoCapture(0)
    classes = get_classes()

    while True:

        if (time.ctime().split()[3][:-3] == '10:31'):
            createNewMlp()
            model3 = joblib.load("MLP2.joblib")
            
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
                            cv.imshow('output', crop)
                            out = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
                            pil = Image.fromarray(out).convert('RGB')

                            image_input = preprocess(pil).unsqueeze(0).to(device)
                            with torch.no_grad():
                                image_features = model2.encode_image(image_input)
                                out = image_features[0].T.cpu().detach().numpy()

                            predict = model3(torch.FloatTensor(out))

                            list = predict.tolist()
                            max_index = list.index(max(list))
                            for value in con2.cursor().execute("SELECT * FROM Buffer WHERE Name IS NULL;"):
                                con2.cursor().execute("UPDATE Buffer SET Name=? WHERE UserId=?",(classes[max_index], value[1]))
                                con2.commit();
                             

        ch = cv.waitKey(5)
        if ch == 27:
            break
    cap.release()

    cv.destroyAllWindows()
    print("Work Finished")
