# run this code to receive the number of documents that contain just narrative, just argument, both narrative and
# argument, and no narrative or argument (first number is the count for human docs, the second is the count for grover
# docs, and the third is the count for davinci docs)

import csv
import re
import os
import sys

import scipy
from scipy.stats import pearsonr
import time
from csv import DictReader
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
with open('1.info.csv', 'r') as f: ## THIS WAS annotation_info, but not in directory?
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
G_Doc_container, H_SE_container, H_Coh_container, H_Doc_container, D_SE_container, D_Coh_container,
D_Doc_container, SE_accounted_for, Coh_accounted_for, doc_counter)

SE_types = ['BOUNDED EVENT (SPECIFIC)', 'BOUNDED EVENT (GENERIC)', 'UNBOUNDED EVENT (SPECIFIC)',
            'BASIC STATE', 'COERCED STATE (SPECIFIC)', 'COERCED STATE (GENERIC)', 'PERFECT COERCED STATE (SPECIFIC)',
            'PERFECT COERCED STATE (GENERIC)', 'GENERIC SENTENCE (HABITUAL)', 'GENERIC SENTENCE (STATIC)',
            'GENERIC SENTENCE (DYNAMIC)', 'GENERALIZING SENTENCE (DYNAMIC)',
            'QUESTION', 'OTHER'] #removed: 'UNBOUNDED EVENT(GENERIC)', 'GENERALIZING SENTENCE (STATIC)'

Coh_types = ['elab', 'temp', 've', 'ce', 'same', 'contr', 'sim', 'attr', 'examp', 'cond', 'rep', 'deg', 'gen']

def annotator_tag(Doc_container): # tags every doc_id with its annotator number

    tagged_dict = {}

    for annotator in Doc_container.keys():

        tagged_dict[annotator] = {}

        for k,v in Doc_container[annotator].items():

            tagged_dict[annotator][str(k) + str(annotator)] = v

    return tagged_dict

# applying the tags to the containers

H_Doc_container = annotator_tag(H_Doc_container)
G_Doc_container = annotator_tag(G_Doc_container)
D_Doc_container = annotator_tag(D_Doc_container)
H_Coh_container = annotator_tag(H_Coh_container)
G_Coh_container = annotator_tag(G_Coh_container)
D_Coh_container = annotator_tag(D_Coh_container)
H_SE_container = annotator_tag(H_SE_container)
G_SE_container = annotator_tag(G_SE_container)
D_SE_container = annotator_tag(D_SE_container)

########################################################

def just_narrative(Doc_container): # counts the number of documents that just contain narratives

    doc_count = 0

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            ratings = Doc_container[annotator][doc_id]

            if ratings[3] != 'NA' and ratings[3] != '0' and \
                    (ratings[12] == 'NA' or ratings[12] == '0'): # finding docs that have some narrative and no argument

                doc_count += 1

    print('Just Narrative Count:', doc_count)

print(just_narrative(H_Doc_container))
print(just_narrative(G_Doc_container))
print(just_narrative(D_Doc_container))

########################################################
########################################################

def just_argument(Doc_container): # counts the number of documents that just contain arguments

    doc_count = 0

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            ratings = Doc_container[annotator][doc_id]

            if ratings[12] != 'NA' and ratings[12] != '0' and \
                    (ratings[3] == 'NA' or ratings[3] == '0'): # finding docs that have some argument and no narrative

                doc_count += 1

    print('Just Argument Count:', doc_count)

print(just_argument(H_Doc_container))
print(just_argument(G_Doc_container))
print(just_argument(D_Doc_container))

########################################################
########################################################

def nar_and_arg(Doc_container): # counts the number of documents that contain both narratives and arguments

    doc_count = 0

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            ratings = Doc_container[annotator][doc_id]

            if ratings[12] != 'NA' and ratings[12] != '0' and \
                    (ratings[3] != 'NA' and ratings[3] != '0'): # finding docs that have some argument and some narrative

                doc_count += 1

    print('Both Narrative and Argument:', doc_count)

print(nar_and_arg(H_Doc_container))
print(nar_and_arg(G_Doc_container))
print(nar_and_arg(D_Doc_container))

###########################################
########################################################

def no_nar_or_arg(Doc_container): # counts the number of documents that contain neither narratives or arguments

    doc_count = 0

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            ratings = Doc_container[annotator][doc_id]

            if (ratings[12] == 'NA' or ratings[12] == '0') and \
                    (ratings[3] == 'NA' or ratings[3] == '0'): # finding docs that have no argument and no narrative

                doc_count += 1

    print('No Narrative or Argument:', doc_count)

print(no_nar_or_arg(H_Doc_container))
print(no_nar_or_arg(G_Doc_container))
print(no_nar_or_arg(D_Doc_container))

###########################################
