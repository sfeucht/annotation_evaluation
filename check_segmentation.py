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
        doc_id = int(f.replace('.txt',''))
        all_docs.append(doc_id)

### then for each document, if its count isn't 1, add to shared_docs list
for doc_id in all_docs:
    if all_docs.count(doc_id) != 1:
        #print(all_docs.count(doc_id))
        shared_docs.append(doc_id)

### remove duplicates from shared_docs list
shared_docs = list(dict.fromkeys(shared_docs))
print(shared_docs)


### For each overlapping doc_id 

