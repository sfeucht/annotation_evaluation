# run this code to calculate the proportion of documents in which the discourse modes present in an original
# document also appear in the corresponding grover or davinci document.

import csv
import re
import os
import sys

import pandas as pd
import scipy
from scipy.stats import pearsonr
from csv import DictReader
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
from clean_up_SE_coh import simplify_all_SE_types
from clean_up_SE_coh import clean_up_coh_rels
from extract_annotations import fill_in_human_grover, fill_in_containers

### determine path to the annotations
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"

### Extract and print out document-annotator assignments
regex = re.compile('[^a-zA-Z]')
annotators = {"0":[],"2":[],"1":[]}
with open('1.info.csv', 'r') as f:
    reader = csv.reader(f)
    for idx,row in enumerate(reader):
        if idx != 0 and len(row) > 1 and "file name" not in row:
            index = re.sub('[^0-9]','',row[0])
            for key in annotators:
                if key.lower() in [i.strip() for i in row[1].split(",")]:
                    annotators[key].append(int(index))

print("***")
print("per annotator count")
print("***")
for annotator in annotators:
    print(annotator)
    print(len(annotators[annotator]))

### EXTRACT THE HUMAN, GROVER, OR DAVINCI SOURCE OF EACH DOCUMENT

# Create lists for keeping track of human and AI generations
h_docs = []
g_docs = []
d_docs = []
fill_in_human_grover(h_docs, g_docs, d_docs)

### Extract Situation Entities, Coherence Relations and Document-level ratings
# from each annotated document

# Create containers
G_SE_container = {"0":{},"2":{},"1":{}}
G_Coh_container = {"0":{},"2":{},"1":{}}
G_Doc_container = {"0":{},"2":{},"1":{}}
H_SE_container = {"0":{},"2":{},"1":{}}
H_Coh_container = {"0":{},"2":{},"1":{}}
H_Doc_container = {"0":{},"2":{},"1":{}}
D_SE_container = {"0":{},"2":{},"1":{}}
D_Coh_container = {"0":{},"2":{},"1":{}}
D_Doc_container = {"0":{},"2":{},"1":{}}
SE_accounted_for = [] # to prevent double-counting of shared documents
Coh_accounted_for = [] # to prevent double-counting of shared documents
doc_counter = 0

doc_counter = fill_in_containers(h_docs, g_docs, d_docs, G_SE_container, G_Coh_container,
G_Doc_container, H_SE_container, H_Coh_container, H_Doc_container, D_SE_container, D_Coh_container, D_Doc_container,
SE_accounted_for, Coh_accounted_for, doc_counter)

#########################################

def corres_calculator(H_container, AI_container, index): # calculates proportion of grover and human documents where
# narrative or argument presence ratings match (index: input 4 for narrative proportion, 12 for argument proportion)

    with open('1.info.csv', 'r') as read_obj:

        csv_dict_reader = DictReader(read_obj)

        matching_pairs = 0
        total_pairs = 0

        for row in csv_dict_reader:

            if row['id'] != 'COLLECTED' and '3_053' not in row['id'] and row['notes'] == '': #filters out section headers, 3_053 docs (which don't have pairs),
                # and any docs that are missing pairs

                for row1 in csv_dict_reader:

                    if row['index'] == row1['index']: # finds a document's pair in the info file

                        doc = []
                        doc1 = []
                        r = 0
                        r1 = 0

                        for annotator in H_container.keys(): # locates original in human doc container

                            if int(row['id']) in H_container[annotator].keys():

                                doc = H_container[annotator][int(row['id'])]

                                r = 'human'

                        for annotator in AI_container.keys(): # locates AI version in grover or davinci doc container

                            if int(row1['id']) in AI_container[annotator].keys():

                                doc1 = AI_container[annotator][int(row1['id'])]

                                r1 = 'ai'

                                if r == 'human' and r1 == 'ai': # to filter out docs with missing pairs

                                    total_pairs += 1

                                    if (doc[index] != '0' and doc1[index] != '0') or (doc[index] == '0' and \
                                        doc1[index] == '0') or (doc[index] == 'NA' and doc1[index] == 'NA'): # checks if discourse modes match

                                        matching_pairs += 1

                        break

        return round((matching_pairs/ total_pairs), 2) # calculates proportion of total pairs that have matching discourse modes

print('Human-Grover Narrative Correspondence:', corres_calculator(H_Doc_container, G_Doc_container, 4))
print('Human-Grover Argument Correspondence:', corres_calculator(H_Doc_container, G_Doc_container, 12))
#based on 168 pairs

print('Human-GPT3 Narrative Correspondence:', corres_calculator(H_Doc_container, D_Doc_container, 4))
print('Human-GPT3 Argument Correspondence:', corres_calculator(H_Doc_container, D_Doc_container, 12))
# based on 24 pairs
