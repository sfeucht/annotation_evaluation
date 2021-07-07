# Counts updated on 12/10/2020
import csv
import re
import os
import sys
from scipy.stats import pearsonr
import time
import matplotlib.pyplot as plt
import numpy as np
import copy

### determine path to the annotations
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"

all_docs = []
shared_docs = []
annotators = {"Sheridan":[],"Muskaan":[],"Kate":[]}

### get all the overlapping doc ids 
### first, put all the doc_ids in a list called all_docs
for annotator in annotators:
    folder = os.listdir("{}/".format(annotator))
    files = [f for f in folder if 'annotation' not in f]
    for f in files:
        doc_id = f.replace('.txt','')
        annotators[annotator].append(doc_id)
        all_docs.append(doc_id)

### then for each document, if its count isn't 1, add to shared_docs list
for doc_id in all_docs:
    if all_docs.count(doc_id) != 1:
        #print(all_docs.count(doc_id))
        shared_docs.append(doc_id)
### remove duplicates from shared_docs list
shared_docs = list(dict.fromkeys(shared_docs))
print(shared_docs, len(shared_docs))

### For each overlapping doc_id, open both docs and compare line by line.
for doc_id in shared_docs:
    ### open both documents
    file1 = 0 
    file2 = 0 
    try:
        for a in annotators:
            folder = os.listdir("{}/".format(a))
            if doc_id in annotators[a]:
                if file1 == 0:
                    file1 = open(path+("{}/".format(a))+doc_id+".txt",'r', encoding="utf-8")
                else:
                    file2 = open(path+("{}/".format(a))+doc_id+".txt",'r', encoding="utf-8")
        assert(file1 != 0 and file2 != 0)


        ### if the number of lines not the same, print doc_id
        file1len = len(file1.readlines())
        file2len = len(file2.readlines())
        if file1len != file2len:
            print("diff no. lines: ", doc_id, file1len, file2len)
            continue

        ### if the number of lines is the same, make sure all the lines are equal, print if not.
        for line1, line2 in zip(file1, file2):
            seg1 = line1.strip().split('##')[0]
            seg2 = line2.strip().split('##')[0]

            ### some annotators got rid of certain spaces, others didn't, so strip all spaces
            seg1strip = seg1.replace(' ', '')
            seg2strip = seg2.replace(' ', '')

            if seg1strip != seg2strip:
                print("lines diff: ", doc_id)
                print("  ", seg1)
                print("  ", seg2)
                break

    except:
        file1.close()
        file2.close()
