import csv
import re
import os
import sys
from scipy.stats import pearsonr
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
annotators = {"Sheridan":[],"Muskaan":[],"Kate":[]}
with open('info.csv','r') as f: ## THIS WAS annotation_info, but not in directory?
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

### Extract the human or Grover source of each document

# Create lists for keeping track of human and Grover generations
h_docs = []
g_docs = []
fill_in_human_grover(h_docs, g_docs)

### Extract Situation Entities, Coherence Relations and Document-level ratings
# from each annotated document

# Create containers
G_SE_container = {"Sheridan":{},"Muskaan":{},"Kate":{}}
G_Coh_container = {"Sheridan":{},"Muskaan":{},"Kate":{}}
G_Doc_container = {"Sheridan":{},"Muskaan":{},"Kate":{}}
H_SE_container = {"Sheridan":{},"Muskaan":{},"Kate":{}}
H_Coh_container = {"Sheridan":{},"Muskaan":{},"Kate":{}}
H_Doc_container = {"Sheridan":{},"Muskaan":{},"Kate":{}}
SE_accounted_for = [] # to prevent double-counting of shared documents
Coh_accounted_for = [] # to prevent double-counting of shared documents
doc_counter = 0

doc_counter = fill_in_containers(h_docs, g_docs, G_SE_container, G_Coh_container, 
G_Doc_container, H_SE_container, H_Coh_container, H_Doc_container, SE_accounted_for,
Coh_accounted_for, doc_counter)


########################################################

### function that gets counts for SE types for all narrative/non-narrative docs 
# gets counts of SE types for _______
def narrative_SE_counts(Doc_container, SE_container, no_narrative_counts, narrative_counts, narrativity_counts):
    for annotator in SE_container.keys():
        for doc_id in SE_container[annotator].keys():
            ratings = Doc_container[annotator][doc_id]
            SE_list = SE_container[annotator][doc_id]

            narrativity_counts[int(ratings[3].strip())] += 1

            # if len(ratings) != 16:
            #     print(doc_id, annotator, len(ratings))

            ## TODO: for each of these things, put in different dictionaries 
            if ratings[3] == '0': # TODO: change ratings[3] to what you're looking at
                for SE in SE_list:
                    if SE.strip() in no_narrative_counts.keys():
                        no_narrative_counts[SE.strip()] += 1
                    else:
                        no_narrative_counts[SE.strip()] = 1
            elif ratings[3] in ['1', '2', '3', '4', '5']: # TODO: compare ratings[3] to something else
                for SE in SE_list:
                    if SE.strip() in narrative_counts.keys():
                        narrative_counts[SE.strip()] += 1 
                    else:
                        narrative_counts[SE.strip()] = 1 


'''
plausibility_0 = 
{'BASIC STATE': 3,
 'COERCED STATE': 2
 ...}

 plausibility_1 = 
{'BASIC STATE': 3,
 'COERCED STATE': 2
 ...}

 plausibility_2 = 
{'BASIC STATE': 3,
 'COERCED STATE': 2
 ...}
'''

### use function to calculate narrativity distribution for human documents
H_no_narrative_counts = {}
H_narrative_counts = {}
H_narrativity_counts = {i:0 for i in range(6)}
narrative_SE_counts(H_Doc_container, H_SE_container, H_no_narrative_counts, H_narrative_counts, H_narrativity_counts)



"""
***0 4 5 0 0 0 0 0 YES ML YES 4 3 1 2 ALGORITHM

0 headline matches
4 style consistent
5 type of news source

0 whether there is narrative
0 plausibility
0 completeness
0 consistency 
0 coverage

YES does this discuss controversial topic
ML what topic
YES is attitude consistent
4 attitude expressed

3 does this contain an argument
1 argument cogent
2 argument effective
ALGORITHM human/algorithm/unclear?
"""
