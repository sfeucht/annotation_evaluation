# run to compare the proportion that each SE type takes up of SE types that are associated with coherent vs incoherent
# coherence relations among Grover documents (uncomment lines 160 and 216 for GPT-3)

import csv
import re
import os
import sys

import scipy
import scipy.stats as st
from scipy.stats import pearsonr
import statistics
from statistics import mean
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
from clean_up_SE_coh import simplify_all_SE_types
from clean_up_SE_coh import clean_up_coh_rels
from extract_annotations import fill_in_human_grover, fill_in_containers

### DETERMINE PATH TO THE ANNOTATIONS
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"

### EXTRACT AND PRINT OUT DOCUMENT-ANNOTATOR ASSIGNMENTS
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

### EXTRACT SITUATION ENTITIES, COHERENCE RELATIONS AND DOCUMENT-LEVEL RATINGS
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
            'QUESTION', 'OTHER']

Coh_types = ['elab', 'temp', 've', 'ce', 'same', 'contr', 'sim', 'attr', 'examp', 'cond', 'deg', 'gen']

def annotator_tag(Doc_container): # tags every doc_id with its annotator number

    tagged_dict = {}

    for annotator in Doc_container.keys():

        tagged_dict[annotator] = {}

        for k,v in Doc_container[annotator].items():

            tagged_dict[annotator][str(k) + str(annotator)] = v

    return tagged_dict

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

def coherent_COH(SE_container, Coh_container): # prints out the proportion of each SE type that is linked to other
# clauses by coherent coherence relations, acts as a baseline for comparison with results from incherent_COH

    SE_list = []

    for annotator in SE_container.keys():
            for SE_id in SE_container[annotator].keys():
                for Coh_id in Coh_container[annotator].keys():

                    Coh_lists = Coh_container[annotator][SE_id]

                    if SE_id == Coh_id: # to match IDs across dicts

                        for list in Coh_lists:

                            if '?' not in list: # filtering out incomplete entries

                                first = [i for i in range(int(list[0]), int(list[1])+1)] # extracting indices from each Coh_list in the Coh_container

                                second = [i for i in range(int(list[2]), int(list[3])+1)]

                                indices = set(first + second) # list of indices provided by each Coh_list (correspond with line numbers in each .txt file)

                                for index in indices: # if given COH is coherent, use indices to find the associated SE type in the SE_container

                                    if(list[4][-1] != 'x' and list[4] != 'rep' and list[4] != 'deg'):

                                        try:

                                            SE_list.append(SE_container[annotator][SE_id][index]) # adding the SE types related to a given coherent COH to a list

                                        except IndexError:
                                            pass

    print('Coherent Coherence Relations')

    for type in SE_types:

        type_list = []

        for SE in SE_list: # counting the number of times that an SE type appears in the SE_list created above

            if type == SE:

                type_list.append(SE)

        print(type, ':', (len(type_list) / len(SE_list))) # calculates the proportion of coherent-COH-associated SEs that each SE type takes up

print(coherent_COH(G_SE_container, G_Coh_container))

#Uncomment for GPT-3 results
#print(coherent_COH(D_SE_container, D_Coh_container))

########################################################
########################################################

def incoherent_COH(SE_container, Coh_container): # prints out the proportion of each SE type that is linked to other
# clauses by incoherent coherence relations

    SE_list = []

    for annotator in SE_container.keys():
            for SE_id in SE_container[annotator].keys():
                for Coh_id in Coh_container[annotator].keys():

                    Coh_lists = Coh_container[annotator][SE_id]

                    if SE_id == Coh_id: # to match IDs across dicts

                        for list in Coh_lists:

                            if '?' not in list: # filtering out incomplete entries

                                first = [i for i in range(int(list[0]), int(list[1])+1)] # extracting indices from each Coh_list in the Coh_container

                                second = [i for i in range(int(list[2]), int(list[3])+1)]

                                indices = set(first + second) # list of indices provided by each Coh_list (correspond with line numbers in each .txt file)

                                for index in indices: # if given COH is incoherent, use indices to find the associated SE type in the SE_container

                                    if(list[4][-1] == 'x' or list[4] == 'rep' or list[4] == 'deg'):

                                        try:

                                            SE_list.append(SE_container[annotator][SE_id][index]) # adding the SE types related to a given incoherent relation to a list

                                        except IndexError:
                                            pass

    print('Incoherent Coherence Relations')

    for type in SE_types:

        type_list = []

        for SE in SE_list: # counting the number of times that an SE type appears in the SE_list created above

            if type == SE:

                type_list.append(SE)

        print(type, ':', (len(type_list) / len(SE_list))) # calculates the proportion of incoherent-COH-associated SEs that each SE type takes up

print(incoherent_COH(G_SE_container, G_Coh_container))

#Uncomment for GPT-3 results
#print(incoherent_COH(D_SE_container, D_Coh_container))

########################################################
