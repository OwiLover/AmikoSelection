import sqlite3
import os
import numpy as np
import torch
import clip
from PIL import Image

def showUrOff(con):

    cur=con.cursor()
    lists = list()
    dest = []
    for value in cur.execute("SELECT Vector, Destination FROM Users"):

        yes = value[0].replace("[", "")
        yes2 = yes.replace("]", "")
        yes3 = yes2.split(" ")
        res=[]
        dest.append(value[1])

        fake = ""
        for sub in yes3:
            if sub!="":
                fake = sub.replace("\n", "")
                res.append(float(fake))

        lists.append(res)
)
    con.close()
    return lists, dest
