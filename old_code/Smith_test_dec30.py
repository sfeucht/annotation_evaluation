# -*- coding: utf-8 -*-

# TODO: adjudicate between annotators. Currently we're taking only one for each
# shared document

# Counts updated on 12/29/2020

### import modules

import csv
import re
import os
import sys
from scipy.stats import pearsonr
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy

### hyperparameters

corr_threshold = 0.2

### determine path to the annotations

file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"

### Extract and print out document-annotator assignments

regex = re.compile('[^a-zA-Z]')
annotators = {"Sheridan":[],"Muskaan":[],"Kate":[]}
with open('annotation_info.csv','r') as f:
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
with open("info.csv","r") as metadata:
    reader = csv.reader(metadata)
    for id_,line in enumerate(reader):
        if id_ != 0 and len(line) > 3:
            if line[3] == "original":
                h_docs.append(int(line[0].strip()))
            elif line[3] == "grover":
                g_docs.append(int(line[0].strip()))

print("***")
print("Total count")
print("***")
print("Human documents")
print(len(h_docs))
print("Grover documents")
print(len(g_docs))

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

for annotator in annotators:

    folder = os.listdir("{}/".format(annotator))

    SE_files = [file for file in folder if 'annotation' not in file]

    # gather all the SE types in each file
    for file in SE_files:

        # get the document ID
        doc_id = int(file.replace('.txt',''))

        if doc_id not in SE_accounted_for:

            doc_counter += 1

            # record the SE types annotated for that document
            with open(path+("{}/".format(annotator))+file,'r', encoding="utf-8") as annotated_doc:

                # initialize container for the SE types
                SE_temp_container = []
                for line in annotated_doc:

                    if line.strip() != "":

                        # save document-level ratings
                        if "***" in line:
                            if doc_id in h_docs:
                                H_Doc_container[annotator][doc_id] = line.strip().replace('***','').split()
                            elif doc_id in g_docs:
                                G_Doc_container[annotator][doc_id] = line.strip().replace('***','').split()
                        else:
                            try:
                                s = line.strip().split('##')[1].split('//')[0]
                                SE_temp_container.append(s.strip('\* '))
                            except:
                                pass

                # record SE ratings
                if doc_id in h_docs:
                    H_SE_container[annotator][doc_id] = SE_temp_container
                elif doc_id in g_docs:
                    G_SE_container[annotator][doc_id] = SE_temp_container
                else:
                    raise Exception("The document index could not be found.")

                SE_accounted_for.append(doc_id)

    # list coherence ratings
    Coh_files = [file for file in folder if 'annotation' in file]

    # for each coherence relations file
    for file in Coh_files:

        # find the id
        doc_id = int(file.replace('.txt-annotation',''))

        if doc_id not in Coh_accounted_for:
            with open(path+("{}/".format(annotator))+file,'r') as annotated_doc:

                # record the coherence relations
                for line in annotated_doc:
                    if line.strip() != "":

                        if doc_id in h_docs:
                            if doc_id not in H_Coh_container[annotator]:
                                H_Coh_container[annotator][doc_id] = [line.strip().split('//')[0].split()]
                            else:
                                H_Coh_container[annotator][doc_id].append(line.strip().split('//')[0].split())
                        elif doc_id in g_docs:
                            if doc_id not in G_Coh_container[annotator]:
                                G_Coh_container[annotator][doc_id] = [line.strip().split('//')[0].split()]
                            else:
                                G_Coh_container[annotator][doc_id].append(line.strip().split('//')[0].split())
                        else:
                            raise Exception("The document index could not be found.")

                Coh_accounted_for.append(doc_id)

print("Annotated document count: ")
print(doc_counter)

print("***")
print("SE types")
print("***")

print("Human")
for element in H_SE_container:
    print(element)
    print(len(H_SE_container[element]))

print("Grover")
for element in G_SE_container:
    print(element)
    print(len(G_SE_container[element]))

print("***")
print("Coherence")
print("***")

print("Human")
for element in H_Coh_container:
    print(element)
    print(len(H_Coh_container[element]))

print("Grover")
for element in G_Coh_container:
    print(element)
    print(len(G_Coh_container[element]))

### Narrativity distribution for human documents

# get counts for SE types for all narrative docs
# weight by amount that is narraive
H_no_narrative_counts = {}
H_narrative_counts = {}
H_narrativity_counts = {i:0 for i in range(6)}

for annotator in H_SE_container.keys():

    for number in H_SE_container[annotator].keys():
        ratings = H_Doc_container[annotator][number]
        SE_list = H_SE_container[annotator][number]

        H_narrativity_counts[int(ratings[3].strip())] += 1

        if ratings[3] == '0': # if no narrative
            for SE in SE_list:
                if SE.strip() in H_no_narrative_counts.keys():
                    H_no_narrative_counts[SE.strip()] += 1
                else:
                    H_no_narrative_counts[SE.strip()] = 1
        elif ratings[3] in ['1', '2', '3', '4', '5']: # if there is narrative
            for SE in SE_list:
                if SE.strip() in H_narrative_counts.keys():
                    H_narrative_counts[SE.strip()] += 1 * (int(ratings[3]) / 5)
                else:
                    H_narrative_counts[SE.strip()] = 1 * (int(ratings[3]) / 5)

print("***")
print("Human Narrativity distribution")
print("***")
print(H_narrativity_counts)

### plot the narrativity distribution

# plt.bar(H_narrativity_counts.keys(), H_narrativity_counts.values(), color='b')
# plt.xticks(list(H_narrativity_counts.keys()),list(H_narrativity_counts.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=35)
# plt.xlabel("Proportion narrative in a document \n(0: none; 5: all)",fontsize=18)
# plt.ylabel("Frequency",fontsize=18)
# plt.title('Human-generated documents',fontsize=20)
# plt.tight_layout()
# plt.show()

### Clean up and rename Situation Entity annotations - Humans

# HYPOTHESIS: narrative documents should have higher avg bounded events and states
# average up for bounded events and states for both dictionaries

H_no_narrative_counts_clean = {
    'BOUNDED EVENT': 0,
    'UNBOUNDED EVENT': 0,
    'BASIC STATE': 0,
    'COERCED STATE': 0,
    'PERFECT COERCED STATE': 0,
    'GENERIC SENTENCE': 0,
    'GENERALIZING SENTENCE': 0,
    'QUESTION': 0,
    'IMPERATIVE': 0,
    'OTHER': 0,
    'NONSENSE': 0
}
H_narrative_counts_clean = copy.deepcopy(H_no_narrative_counts_clean)

def clean_up_SE_types(old_dict, clean_dict):
    for SE_type in old_dict.keys():
        if SE_type in clean_dict.keys():
            clean_dict[SE_type] += old_dict[SE_type]
        elif SE_type in ['BOUNDED EVENT (SPECIFIC)', 'BOUNDED EVENT (GENERIC)', 'BOUNDED EVENT (SPECIFIC_']:
            clean_dict['BOUNDED EVENT'] += old_dict[SE_type]
        elif SE_type in ['UNBOUNDED EVENT (SPECIFIC)', 'UNBOUNDED EVENT (GENERIC)', 'UNBOUNDED EVENT (GENERIC0', 'NBOUNDED EVENT (SPECIFIC)']:
            clean_dict['UNBOUNDED EVENT'] += old_dict[SE_type]
        elif SE_type in ['COERCED STATE (SPECIFIC)', 'COERCED STATE (GENERIC)', 'COERCED STATE (STATIC)']:
            clean_dict['COERCED STATE'] += old_dict[SE_type]
        elif SE_type in ['PERFECT COERCED STATE (SPECIFIC)', 'PERFECT COERCED STATE (GENERIC)']:
            clean_dict['PERFECT COERCED STATE'] += old_dict[SE_type]
        elif SE_type in ['GENERIC SENTENCE (STATIC)', 'GENERIC SENTENCE (SATIC)', 'GNERIC SENTENCE (STATIC)', 'GNERIC SENTENCE (STATIC0', 'GENERIC SENTENCE (STATAIC)', 'GENERIC STATE (STATIC)', 'GENERIC SENTENCE (DYNAMIC)', 'GENERIC SENTENCE (HABITUAL)']:
            clean_dict['GENERIC SENTENCE'] += old_dict[SE_type]
        elif SE_type in ['GENERALIZING SENTENCE (DYNAMIC)', 'GENERALIZING SENTENCE (STATIVE)']:
            clean_dict['GENERALIZING SENTENCE'] += old_dict[SE_type]
        elif SE_type in ['BASIC SENTENCE', 'BAISC STATE', 'BASIC STATEA', 'BASIC SETATE']:
            clean_dict['BASIC STATE'] += old_dict[SE_type]
        elif SE_type in ['OHTER', 'ITGER', 'OTEHR', 'OTHE R']:
            clean_dict['OTHER'] += old_dict[SE_type]

    for k in clean_dict.keys():
        clean_dict[k] /= sum(old_dict.values())

# count up for non narrative texts
clean_up_SE_types(H_no_narrative_counts, H_no_narrative_counts_clean)
clean_up_SE_types(H_narrative_counts, H_narrative_counts_clean)

### Plot Situation Entity distribution divided by narrativity - Human

# max_num = np.max([max(list(H_no_narrative_counts_clean.values())),max(list(H_narrative_counts_clean.values()))])
# plt.bar(H_no_narrative_counts_clean.keys(), H_no_narrative_counts_clean.values(), color='b')
# plt.xticks(list(H_no_narrative_counts_clean.keys()),list(H_no_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.3)
# plt.xlabel("Situation Entity Type",fontsize=18)
# plt.ylabel("Proportion of non-narrative SEs",fontsize=18)
# plt.title('Docs Without Narratives - Human',fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(H_narrative_counts_clean.keys(), H_narrative_counts_clean.values(), color='b')
# plt.xticks(list(H_narrative_counts_clean.keys()),list(H_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.3)
# plt.xlabel("Situation Entity Type",fontsize=18)
# plt.ylabel("Proportion of narrative SEs",fontsize=18)
# plt.title('Docs With Narratives - Human',fontsize=20)
# plt.tight_layout()
# plt.show()

### Narrativity distribution for Grover documents

# get counts for SE types for all narrative docs
# weight by amount that is narraive
G_no_narrative_counts = {}
G_narrative_counts = {}
G_narrativity_counts = {i:0 for i in range(6)}

for annotator in G_SE_container.keys():

    for number in G_SE_container[annotator].keys():
        ratings = G_Doc_container[annotator][number]
        SE_list = G_SE_container[annotator][number]

        G_narrativity_counts[int(ratings[3].strip())] += 1

        if ratings[3] == '0': # if no narrative
            for SE in SE_list:
                if SE.strip() in G_no_narrative_counts.keys():
                    G_no_narrative_counts[SE.strip()] += 1
                else:
                    G_no_narrative_counts[SE.strip()] = 1
        elif ratings[3] in ['1', '2', '3', '4', '5']: # if there is narrative
            for SE in SE_list:
                if SE.strip() in G_narrative_counts.keys():
                    G_narrative_counts[SE.strip()] += 1 * (int(ratings[3]) / 5)
                else:
                    G_narrative_counts[SE.strip()] = 1 * (int(ratings[3]) / 5)

print("***")
print("Grover Narrativity distribution")
print("***")
print(G_narrativity_counts)

### plot the narrativity distribution

# plt.bar(G_narrativity_counts.keys(), G_narrativity_counts.values(), color='g')
# plt.xticks(list(G_narrativity_counts.keys()),list(G_narrativity_counts.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=35)
# plt.xlabel("Proportion narrative in a document \n(0: none; 5: all)",fontsize=18)
# plt.ylabel("Frequency",fontsize=18)
# plt.title('Grover-generated documents',fontsize=20)
# plt.tight_layout()
# plt.show()

### Clean up and rename Situation Entity annotations - Humans

# HYPOTHESIS: narrative documents should have higher avg bounded events and states
# average up for bounded events and states for both dictionaries

G_no_narrative_counts_clean = {
    'BOUNDED EVENT': 0,
    'UNBOUNDED EVENT': 0,
    'BASIC STATE': 0,
    'COERCED STATE': 0,
    'PERFECT COERCED STATE': 0,
    'GENERIC SENTENCE': 0,
    'GENERALIZING SENTENCE': 0,
    'QUESTION': 0,
    'IMPERATIVE': 0,
    'OTHER': 0,
    'NONSENSE': 0
}
G_narrative_counts_clean = copy.deepcopy(G_no_narrative_counts_clean)

# count up for non narrative texts
clean_up_SE_types(G_no_narrative_counts, G_no_narrative_counts_clean)
clean_up_SE_types(G_narrative_counts, G_narrative_counts_clean)

### Plot Situation Entity distribution divided by narrativity - Grover

# max_num = np.max([max(list(G_no_narrative_counts_clean.values())),max(list(G_narrative_counts_clean.values()))])
# plt.bar(G_no_narrative_counts_clean.keys(), G_no_narrative_counts_clean.values(), color='g')
# plt.xticks(list(G_no_narrative_counts_clean.keys()),list(G_no_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.3)
# plt.xlabel("Situation Entity Type",fontsize=18)
# plt.ylabel("Proportion of non-narrative SEs",fontsize=18)
# plt.title('Docs Without Narratives - Grover',fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(G_narrative_counts_clean.keys(), G_narrative_counts_clean.values(), color='g')
# plt.xticks(list(G_narrative_counts_clean.keys()),list(G_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.3)
# plt.xlabel("Situation Entity Type",fontsize=18)
# plt.ylabel("Proportion of narrative SEs",fontsize=18)
# plt.title('Docs With Narratives - Grover',fontsize=20)
# plt.tight_layout()
# plt.show()

###########################################################################

### SE Analysis for Arguments ###

### Human-generated documents
H_no_argument_counts = {}
H_argument_counts = {}
H_argumentation = {"argument":0,"no argument":0}

for annotator in H_SE_container.keys():

    for number in H_SE_container[annotator].keys():

        ratings = H_Doc_container[annotator][number]
        SE_list = H_SE_container[annotator][number]

        if ratings[12] == '0': # if no argument

            H_argumentation["no argument"] += 1

            for SE in SE_list:
                if SE.strip() in H_no_argument_counts.keys():
                    H_no_argument_counts[SE.strip()] += 1
                else:
                    H_no_argument_counts[SE.strip()] = 1
        elif ratings[12] in ['1', '2', '3']: # if there is argument

            H_argumentation["argument"] += 1

            for SE in SE_list:
                if SE.strip() in H_argument_counts.keys():
                    H_argument_counts[SE.strip()] += 1
                else:
                    H_argument_counts[SE.strip()] = 1

### Clean up and rename Situation Entity annotations - Humans

H_no_argument_counts_clean = {
    'BOUNDED EVENT': 0,
    'UNBOUNDED EVENT': 0,
    'BASIC STATE': 0,
    'COERCED STATE': 0,
    'PERFECT COERCED STATE': 0,
    'GENERIC SENTENCE': 0,
    'GENERALIZING SENTENCE': 0,
    'QUESTION': 0,
    'IMPERATIVE': 0,
    'OTHER': 0,
    'NONSENSE': 0
}
H_argument_counts_clean = copy.deepcopy(H_no_argument_counts_clean)

# clean up for both dicts
clean_up_SE_types(H_no_argument_counts, H_no_argument_counts_clean)
clean_up_SE_types(H_argument_counts, H_argument_counts_clean)

print("***")
print("Human argumentation distribution")
print("***")
print(H_argumentation)

### Plot Situation Entity distribution divided by argumentation - Human

# plt.bar(H_no_argument_counts_clean.keys(), H_no_argument_counts_clean.values(), color='b')
# plt.xticks(list(H_no_argument_counts_clean.keys()),list(H_no_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.3)
# plt.xlabel("Situation Entity Type", fontsize=18)
# plt.ylabel("Proportion of SEs", fontsize=18)
# plt.title('Docs without Arguments - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(H_argument_counts_clean.keys(), H_argument_counts_clean.values(), color='b')
# plt.xticks(list(H_argument_counts_clean.keys()),list(H_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.3) #ymax=max_num+5
# plt.xlabel("Situation Entity Type",fontsize=18)
# plt.ylabel("Proportion of SEs",fontsize=18) 
# plt.title('Docs with Arguments - Human',fontsize=20)
# plt.tight_layout()
# plt.show()

### Grover-generated documents
G_no_argument_counts = {}
G_argument_counts = {}
G_argumentation = {"argument":0,"no argument":0}

for annotator in G_SE_container.keys():

    for number in G_SE_container[annotator].keys():

        ratings = G_Doc_container[annotator][number]
        SE_list = G_SE_container[annotator][number]

        if ratings[12] == '0': # if no argument

            G_argumentation["no argument"] += 1

            for SE in SE_list:
                if SE.strip() in G_no_argument_counts.keys():
                    G_no_argument_counts[SE.strip()] += 1
                else:
                    G_no_argument_counts[SE.strip()] = 1
        elif ratings[12] in ['1', '2', '3']: # if there is argument

            G_argumentation["argument"] += 1

            for SE in SE_list:
                if SE.strip() in G_argument_counts.keys():
                    G_argument_counts[SE.strip()] += 1
                else:
                    G_argument_counts[SE.strip()] = 1

### Clean up and rename Situation Entity annotations - Grover

G_no_argument_counts_clean = {
    'BOUNDED EVENT': 0,
    'UNBOUNDED EVENT': 0,
    'BASIC STATE': 0,
    'COERCED STATE': 0,
    'PERFECT COERCED STATE': 0,
    'GENERIC SENTENCE': 0,
    'GENERALIZING SENTENCE': 0,
    'QUESTION': 0,
    'IMPERATIVE': 0,
    'OTHER': 0,
    'NONSENSE': 0
}
G_argument_counts_clean = copy.deepcopy(G_no_argument_counts_clean)

# clean up for both dicts
clean_up_SE_types(G_no_argument_counts, G_no_argument_counts_clean)
clean_up_SE_types(G_argument_counts, G_argument_counts_clean)

print("***")
print("Grover argumentation distribution")
print("***")
print(G_argumentation)

### Plot Situation Entity distribution divided by argumentation - Grover

# plt.bar(G_no_argument_counts_clean.keys(), G_no_argument_counts_clean.values(), color='g')
# plt.xticks(list(G_no_argument_counts_clean.keys()),list(G_no_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.3)
# plt.xlabel("Situation Entity Type", fontsize=18)
# plt.ylabel("Proportion of SEs", fontsize=18)
# plt.title('Docs without Arguments - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(G_argument_counts_clean.keys(), G_argument_counts_clean.values(), color='g')
# plt.xticks(list(G_argument_counts_clean.keys()),list(G_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.3) #ymax=max_num+5
# plt.xlabel("Situation Entity Type", fontsize=18)
# plt.ylabel("Proportion of SEs", fontsize=18)
# plt.title('Docs with Arguments - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

##############################################################################

### Coherence Relations ###

### Extract coherence relations from annotated documents divided by narrativtiy
# and argumentation

H_Coh_no_narrative_counts = {}
H_Coh_narrative_counts = {}
H_Coh_no_argument_counts = {}
H_Coh_argument_counts = {}

### Human counts
for annotator in H_Coh_container:
    for number in H_Coh_container[annotator]:
        ratings = H_Doc_container[annotator][number]

        if ratings[3] != '0': # narrative
            for element in H_Coh_container[annotator][number]:
                if element[4] not in H_Coh_narrative_counts:
                    H_Coh_narrative_counts[element[4]] = 1
                else:
                    H_Coh_narrative_counts[element[4]] += 1
        else:
            for element in H_Coh_container[annotator][number]:
                if element[4] not in H_Coh_no_narrative_counts:
                    H_Coh_no_narrative_counts[element[4]] = 1
                else:
                    H_Coh_no_narrative_counts[element[4]] += 1

        if ratings[12] != '0': # argument
            for element in H_Coh_container[annotator][number]:
                if element[4] not in H_Coh_argument_counts:
                    H_Coh_argument_counts[element[4]] = 1
                else:
                    H_Coh_argument_counts[element[4]] += 1
        else:
            for element in H_Coh_container[annotator][number]:
                if element[4] not in H_Coh_no_argument_counts:
                    H_Coh_no_argument_counts[element[4]] = 1
                else:
                    H_Coh_no_argument_counts[element[4]] += 1

### Grover counts
G_Coh_no_narrative_counts = {}
G_Coh_narrative_counts = {}
G_Coh_no_argument_counts = {}
G_Coh_argument_counts = {}

for annotator in G_Coh_container:
    for number in G_Coh_container[annotator]:

        if number == 51020131204:
            del G_Coh_container[annotator][number][0]

        ratings = G_Doc_container[annotator][number]

        if ratings[3] != '0': # narrative
            for id_,element in enumerate(G_Coh_container[annotator][number]):
                if element[4] not in G_Coh_narrative_counts:
                    G_Coh_narrative_counts[element[4]] = 1
                else:
                    G_Coh_narrative_counts[element[4]] += 1
        else:
            for element in G_Coh_container[annotator][number]:
                if element[4] not in G_Coh_no_narrative_counts:
                    G_Coh_no_narrative_counts[element[4]] = 1
                else:
                    G_Coh_no_narrative_counts[element[4]] += 1

        if ratings[12] != '0': # argument
            for element in G_Coh_container[annotator][number]:
                if element[4] not in G_Coh_argument_counts:
                    G_Coh_argument_counts[element[4]] = 1
                else:
                    G_Coh_argument_counts[element[4]] += 1
        else:
            for element in G_Coh_container[annotator][number]:
                if element[4] not in G_Coh_no_argument_counts:
                    G_Coh_no_argument_counts[element[4]] = 1
                else:
                    G_Coh_no_argument_counts[element[4]] += 1

### Clean up and rename coherence relations separated by human origin, narrativity,
# and argumentation

def clean_up_coh_rels(old_dict,clean_dict):
    for relation in old_dict.keys():
        if relation in clean_dict.keys():
            clean_dict[relation] += old_dict[relation]
        elif relation in ['ce', 'cex', 'cew', 'cer']:
            clean_dict['Cause/effect'] += old_dict[relation]
        elif relation in ['elab', 'elabx', 'ealb', 'elav', 'elabl']:
            clean_dict['Elaboration'] += old_dict[relation]
        elif relation in ['same', 'samex']:
            clean_dict['Same'] += old_dict[relation]
        elif relation in ['attr', 'attrx', 'attrm']:
            clean_dict['Attribution'] += old_dict[relation]
        elif relation in ['deg', 'degenerate', 'mal']:
            clean_dict['Degenerate'] += old_dict[relation]
        elif relation in ['sim', 'simx']:
            clean_dict['Similarity'] += old_dict[relation]
        elif relation in ['contr', 'contrx']:
            clean_dict['Contrast'] += old_dict[relation]
        elif relation in ['temp', 'tempx']:
            clean_dict['Temporal sequence'] += old_dict[relation]
        elif relation in ['ve', 'vex']:
            clean_dict['Violated expectation'] += old_dict[relation]
        elif relation in ['examp', 'exampx']:
            clean_dict['Example'] += old_dict[relation]
        elif relation in ['cond','condx']:
            clean_dict['Condition'] += old_dict[relation]
        elif relation in ['gen', 'genx']:
            clean_dict['Generalization'] += old_dict[relation]
        elif relation in ['rep']:
            clean_dict['Repetition'] += old_dict[relation]

H_Coh_no_narrative_counts_clean = {
        'Cause/effect': 0,
        'Elaboration': 0,
        'Same': 0,
        'Attribution': 0,
        'Degenerate': 0,
        'Similarity': 0,
        'Contrast': 0,
        'Temporal sequence': 0,
        'Violated expectation': 0,
        'Example': 0,
        'Condition': 0,
        'Generalization': 0,
        'Repetition': 0}
H_Coh_narrative_counts_clean = copy.deepcopy(H_Coh_no_narrative_counts_clean)
H_Coh_no_argument_counts_clean = copy.deepcopy(H_Coh_no_narrative_counts_clean)
H_Coh_argument_counts_clean = copy.deepcopy(H_Coh_no_narrative_counts_clean)
G_Coh_no_narrative_counts_clean = copy.deepcopy(H_Coh_no_narrative_counts_clean)
G_Coh_narrative_counts_clean = copy.deepcopy(H_Coh_no_narrative_counts_clean)
G_Coh_no_argument_counts_clean = copy.deepcopy(H_Coh_no_narrative_counts_clean)
G_Coh_argument_counts_clean = copy.deepcopy(H_Coh_no_narrative_counts_clean)

clean_up_coh_rels(H_Coh_no_narrative_counts,H_Coh_no_narrative_counts_clean)
clean_up_coh_rels(H_Coh_narrative_counts,H_Coh_narrative_counts_clean)
clean_up_coh_rels(H_Coh_no_argument_counts,H_Coh_no_argument_counts_clean)
clean_up_coh_rels(H_Coh_argument_counts,H_Coh_argument_counts_clean)
clean_up_coh_rels(G_Coh_no_narrative_counts,G_Coh_no_narrative_counts_clean)
clean_up_coh_rels(G_Coh_narrative_counts,G_Coh_narrative_counts_clean)
clean_up_coh_rels(G_Coh_no_argument_counts,G_Coh_no_argument_counts_clean)
clean_up_coh_rels(G_Coh_argument_counts,G_Coh_argument_counts_clean)

### Plot the distribution of coherence relations divided by narrativity - Human

# print(H_Coh_no_narrative_counts_clean)

# plt.bar(H_Coh_no_narrative_counts_clean.keys(), H_Coh_no_narrative_counts_clean.values(), color='b')
# plt.xticks(list(H_Coh_no_narrative_counts_clean.keys()),list(H_Coh_no_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=200)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title('Docs without Narratives - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

# print(H_Coh_narrative_counts_clean)

# plt.bar(H_Coh_narrative_counts_clean.keys(), H_Coh_narrative_counts_clean.values(), color='b')
# plt.xticks(list(H_Coh_narrative_counts_clean.keys()),list(H_Coh_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=200)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title('Docs with Narratives - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

### Plot the distribution of coherence relations divided by narrativity - Grover

# plt.bar(G_Coh_no_narrative_counts_clean.keys(), G_Coh_no_narrative_counts_clean.values(), color='g')
# plt.xticks(list(G_Coh_no_narrative_counts_clean.keys()),list(G_Coh_no_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=200)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title('Docs without Narratives - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

# print(G_Coh_no_narrative_counts_clean)

# plt.bar(G_Coh_narrative_counts_clean.keys(), G_Coh_narrative_counts_clean.values(), color='g')
# plt.xticks(list(G_Coh_narrative_counts_clean.keys()),list(G_Coh_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=200)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title('Docs with Narratives - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

# print(G_Coh_narrative_counts_clean)

### Plot the distribution of coherence relations divided by argumentation - Human

# print(H_Coh_no_argument_counts_clean)

# plt.bar(H_Coh_no_argument_counts_clean.keys(), H_Coh_no_argument_counts_clean.values(), color='b')
# plt.xticks(list(H_Coh_no_argument_counts_clean.keys()),list(H_Coh_no_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=210)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title('Docs without Arguments - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

# print(H_Coh_argument_counts_clean)

# plt.bar(H_Coh_argument_counts_clean.keys(), H_Coh_argument_counts_clean.values(), color='b')
# plt.xticks(list(H_Coh_argument_counts_clean.keys()),list(H_Coh_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=210)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title('Docs with Arguments - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

### Plot the distribution of coherence relations divided by argumentation - Grover

# print(G_Coh_no_argument_counts_clean)

# plt.bar(G_Coh_no_argument_counts_clean.keys(), G_Coh_no_argument_counts_clean.values(), color='g')
# plt.xticks(list(G_Coh_no_argument_counts_clean.keys()),list(G_Coh_no_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=220)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title('Docs without Arguments - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

# print(G_Coh_argument_counts_clean)

# plt.bar(G_Coh_argument_counts_clean.keys(), G_Coh_argument_counts_clean.values(), color='g')
# plt.xticks(list(G_Coh_argument_counts_clean.keys()),list(G_Coh_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=220)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title('Docs with Arguments - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

### Normalize coherence relation rates to show proportions

for k in H_Coh_no_narrative_counts_clean:
    H_Coh_no_narrative_counts_clean[k] /= sum(H_Coh_no_narrative_counts.values())
for k in H_Coh_narrative_counts_clean:
    H_Coh_narrative_counts_clean[k] /= sum(H_Coh_narrative_counts.values())
for k in H_Coh_no_argument_counts_clean:
    H_Coh_no_argument_counts_clean[k] /= sum(H_Coh_no_argument_counts.values())
for k in H_Coh_argument_counts_clean:
    H_Coh_argument_counts_clean[k] /= sum(H_Coh_argument_counts.values())
for k in G_Coh_no_narrative_counts_clean:
    G_Coh_no_narrative_counts_clean[k] /= sum(G_Coh_no_narrative_counts.values())
for k in G_Coh_narrative_counts_clean:
    G_Coh_narrative_counts_clean[k] /= sum(G_Coh_narrative_counts.values())
for k in G_Coh_no_argument_counts_clean:
    G_Coh_no_argument_counts_clean[k] /= sum(G_Coh_no_argument_counts.values())
for k in G_Coh_argument_counts_clean:
    G_Coh_argument_counts_clean[k] /= sum(G_Coh_argument_counts.values())

### Plot the proportion of coherence relations divided by narrativity - Human

# print(H_Coh_no_narrative_counts_clean)

# plt.bar(H_Coh_no_narrative_counts_clean.keys(), H_Coh_no_narrative_counts_clean.values(), color='b')
# plt.xticks(list(H_Coh_no_narrative_counts_clean.keys()),list(H_Coh_no_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.15)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Proportion of relations", fontsize=18)
# plt.title('Docs without Narratives - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

# print(H_Coh_narrative_counts_clean)

# plt.bar(H_Coh_narrative_counts_clean.keys(), H_Coh_narrative_counts_clean.values(), color='b')
# plt.xticks(list(H_Coh_narrative_counts_clean.keys()),list(H_Coh_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.15)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Proportion of relations", fontsize=18)
# plt.title('Docs with Narratives - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

### Plot the proportion of coherence relations divided by narrativity - Grover

# plt.bar(G_Coh_no_narrative_counts_clean.keys(), G_Coh_no_narrative_counts_clean.values(), color='g')
# plt.xticks(list(G_Coh_no_narrative_counts_clean.keys()),list(G_Coh_no_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.15)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Proportion of relations", fontsize=18)
# plt.title('Docs without Narratives - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(G_Coh_narrative_counts_clean.keys(), G_Coh_narrative_counts_clean.values(), color='g')
# plt.xticks(list(G_Coh_narrative_counts_clean.keys()),list(G_Coh_narrative_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.15)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Proportion of relations", fontsize=18)
# plt.title('Docs with Narratives - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

### Plot the proportion of coherence relations divided by argumentation - Human

# plt.bar(H_Coh_no_argument_counts_clean.keys(), H_Coh_no_argument_counts_clean.values(), color='b')
# plt.xticks(list(H_Coh_no_argument_counts_clean.keys()),list(H_Coh_no_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.15)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Proportion of relations", fontsize=18)
# plt.title('Docs without Arguments - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(H_Coh_argument_counts_clean.keys(), H_Coh_argument_counts_clean.values(), color='b')
# plt.xticks(list(H_Coh_argument_counts_clean.keys()),list(H_Coh_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.15)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Proportion of relations", fontsize=18)
# plt.title('Docs with Arguments - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

### Plot the proportion of coherence relations divided by argumentation - Grover

# plt.bar(G_Coh_no_argument_counts_clean.keys(), G_Coh_no_argument_counts_clean.values(), color='g')
# plt.xticks(list(G_Coh_no_argument_counts_clean.keys()),list(G_Coh_no_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.15)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Proportion of relations", fontsize=18)
# plt.title('Docs without Arguments - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(G_Coh_argument_counts_clean.keys(), G_Coh_argument_counts_clean.values(), color='g')
# plt.xticks(list(G_Coh_argument_counts_clean.keys()),list(G_Coh_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.15)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Proportion of relations", fontsize=18)
# plt.title('Docs with Arguments - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

### Average lengths of coherence relations, divided by narrativity & arguments

### ADAM's Thematic Unity Hypothesis: documents marked as containing a `narrative`
# should have more longer-range coherence relations, suggesting they are talking
# about a central topic

### Clean up and rename coherence relations - narrativity, human

H_narrative_lengths = {}
H_no_narrative_lengths = {}

for annotator in H_Coh_container:
    for number in H_Coh_container[annotator]:
        for relation in H_Coh_container[annotator][number]:
            try:
                tmp = [int(i) for i in relation[0:4]]
                min = np.min(tmp)
                max = np.max(tmp)
                if H_Doc_container[annotator][number][3] != '0': # if narrative
                    if relation[4] in H_narrative_lengths:
                        H_narrative_lengths[relation[4]].append(max-min)
                    else:
                        diff = max-min
                        H_narrative_lengths[relation[4]] = [diff]
                else:
                    if relation[4] in H_no_narrative_lengths:
                        H_no_narrative_lengths[relation[4]].append(max-min)
                    else:
                        diff = max-min
                        H_no_narrative_lengths[relation[4]] = [diff]
            except: # because sometimes there's ? instead of a segment number
                pass

H_Coh_narrative_lengths_clean = {
        'Cause/effect': [],
        'Elaboration': [],
        'Same': [],
        'Attribution': [],
        'Degenerate': [],
        'Similarity': [],
        'Contrast': [],
        'Temporal sequence': [],
        'Violated expectation': [],
        'Example': [],
        'Condition': [],
        'Generalization': [],
        'Repetition': []
        }

H_Coh_no_narrative_lengths_clean = copy.deepcopy(H_Coh_narrative_lengths_clean)

def clean_up_coh_lengths(old_dict,clean_dict):
    for relation in old_dict.keys():
        if relation in ['ce', 'cex', 'cew', 'cer']:
            for element in old_dict[relation]:
                clean_dict['Cause/effect'].append(element)
        elif relation in ['elab', 'elabx', 'ealb', 'elav', 'elabl']:
            for element in old_dict[relation]:
                clean_dict['Elaboration'].append(element)
        elif relation in ['same', 'samex']:
            for element in old_dict[relation]:
                clean_dict['Same'].append(element)
        elif relation in ['attr', 'attrx', 'attrm']:
            for element in old_dict[relation]:
                clean_dict['Attribution'].append(element)
        elif relation in ['deg', 'degenerate', 'mal']:
            for element in old_dict[relation]:
                clean_dict['Degenerate'].append(element)
        elif relation in ['sim', 'simx']:
            for element in old_dict[relation]:
                clean_dict['Similarity'].append(element)
        elif relation in ['contr', 'contrx']:
            for element in old_dict[relation]:
                clean_dict['Contrast'].append(element)
        elif relation in ['temp', 'tempx']:
            for element in old_dict[relation]:
                clean_dict['Temporal sequence'].append(element)
        elif relation in ['ve', 'vex']:
            for element in old_dict[relation]:
                clean_dict['Violated expectation'].append(element)
        elif relation in ['examp', 'exampx']:
            for element in old_dict[relation]:
                clean_dict['Example'].append(element)
        elif relation in ['cond','condx']:
            for element in old_dict[relation]:
                clean_dict['Condition'].append(element)
        elif relation in ['gen', 'genx']:
            for element in old_dict[relation]:
                clean_dict['Generalization'].append(element)
        elif relation in ['rep']:
            for element in old_dict[relation]:
                clean_dict['Repetition'].append(element)

clean_up_coh_lengths(H_narrative_lengths,H_Coh_narrative_lengths_clean)
clean_up_coh_lengths(H_no_narrative_lengths,H_Coh_no_narrative_lengths_clean)

### Calculate average coherence relation lengths - narrativity, human

H_Coh_narrative_avg_lengths = {}
for relation in H_Coh_narrative_lengths_clean:
    H_Coh_narrative_avg_lengths[relation] = np.mean(H_Coh_narrative_lengths_clean[relation])

H_Coh_no_narrative_avg_lengths = {}
for relation in H_Coh_no_narrative_lengths_clean:
    H_Coh_no_narrative_avg_lengths[relation] = np.mean(H_Coh_no_narrative_lengths_clean[relation])

### Plot average coherence relation lengths divided by narrativity - Human

# plt.bar(H_Coh_narrative_avg_lengths.keys(), H_Coh_narrative_avg_lengths.values(), color='b')
# plt.xticks(list(H_Coh_no_argument_counts_clean.keys()),list(H_Coh_no_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=20)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Average length", fontsize=18)
# plt.title('Docs with narrative - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(H_Coh_no_narrative_avg_lengths.keys(), H_Coh_no_narrative_avg_lengths.values(), color='b')
# plt.xticks(list(H_Coh_no_narrative_avg_lengths.keys()),list(H_Coh_no_narrative_avg_lengths.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=20)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Average length", fontsize=18)
# plt.title('Docs without narrative - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

### Clean up and rename coherence relations - narrativity, Grover

G_narrative_lengths = {}
G_no_narrative_lengths = {}

for annotator in G_Coh_container:
    for number in G_Coh_container[annotator]:
        for relation in G_Coh_container[annotator][number]:
            try:
                tmp = [int(i) for i in relation[0:4]]
                min = np.min(tmp)
                max = np.max(tmp)
                if G_Doc_container[annotator][number][3] != '0': # if narrative
                    if relation[4] in G_narrative_lengths:
                        G_narrative_lengths[relation[4]].append(max-min)
                    else:
                        diff = max-min
                        G_narrative_lengths[relation[4]] = [diff]
                else:
                    if relation[4] in G_no_narrative_lengths:
                        G_no_narrative_lengths[relation[4]].append(max-min)
                    else:
                        diff = max-min
                        G_no_narrative_lengths[relation[4]] = [diff]
            except: # because sometimes there's ? instead of a segment number
                pass

G_Coh_narrative_lengths_clean = {
        'Cause/effect': [],
        'Elaboration': [],
        'Same': [],
        'Attribution': [],
        'Degenerate': [],
        'Similarity': [],
        'Contrast': [],
        'Temporal sequence': [],
        'Violated expectation': [],
        'Example': [],
        'Condition': [],
        'Generalization': [],
        'Repetition': []
        }

G_Coh_no_narrative_lengths_clean = copy.deepcopy(G_Coh_narrative_lengths_clean)

clean_up_coh_lengths(G_narrative_lengths,G_Coh_narrative_lengths_clean)
clean_up_coh_lengths(G_no_narrative_lengths,G_Coh_no_narrative_lengths_clean)

### Calculate average coherence relation lengths - narrativity, Grover

G_Coh_narrative_avg_lengths = {}
for relation in G_Coh_narrative_lengths_clean:
    G_Coh_narrative_avg_lengths[relation] = np.mean(G_Coh_narrative_lengths_clean[relation])

G_Coh_no_narrative_avg_lengths = {}
for relation in G_Coh_no_narrative_lengths_clean:
    G_Coh_no_narrative_avg_lengths[relation] = np.mean(G_Coh_no_narrative_lengths_clean[relation])

### Plot average coherence relation lengths divided by narrativity - Grover

# plt.bar(G_Coh_narrative_avg_lengths.keys(), G_Coh_narrative_avg_lengths.values(), color='g')
# plt.xticks(list(G_Coh_no_argument_counts_clean.keys()),list(G_Coh_no_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=20)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Average length", fontsize=18)
# plt.title('Docs with narrative - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(G_Coh_no_narrative_avg_lengths.keys(), G_Coh_no_narrative_avg_lengths.values(), color='g')
# plt.xticks(list(G_Coh_no_narrative_avg_lengths.keys()),list(G_Coh_no_narrative_avg_lengths.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=20)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Average length", fontsize=18)
# plt.title('Docs without narrative - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

### Clean up and rename coherence relations - argumentation, human

H_argument_lengths = {}
H_no_argument_lengths = {}

for annotator in H_Coh_container:
    for number in H_Coh_container[annotator]:
        for relation in H_Coh_container[annotator][number]:
            try:
                tmp = [int(i) for i in relation[0:4]]
                min = np.min(tmp)
                max = np.max(tmp)
                if H_Doc_container[annotator][number][12] != '0': # if argument
                    if relation[4] in H_argument_lengths:
                        H_argument_lengths[relation[4]].append(max-min)
                    else:
                        diff = max-min
                        H_argument_lengths[relation[4]] = [diff]
                else:
                    if relation[4] in H_no_argument_lengths:
                        H_no_argument_lengths[relation[4]].append(max-min)
                    else:
                        diff = max-min
                        H_no_argument_lengths[relation[4]] = [diff]
            except: # because sometimes there's ? instead of a segment number
                pass

H_Coh_argument_lengths_clean = {
        'Cause/effect': [],
        'Elaboration': [],
        'Same': [],
        'Attribution': [],
        'Degenerate': [],
        'Similarity': [],
        'Contrast': [],
        'Temporal sequence': [],
        'Violated expectation': [],
        'Example': [],
        'Condition': [],
        'Generalization': [],
        'Repetition': []
        }

H_Coh_no_argument_lengths_clean = copy.deepcopy(H_Coh_argument_lengths_clean)

clean_up_coh_lengths(H_argument_lengths,H_Coh_argument_lengths_clean)
clean_up_coh_lengths(H_no_argument_lengths,H_Coh_no_argument_lengths_clean)

### Calculate average coherence relation lengths - argumentation, human

H_Coh_argument_avg_lengths = {}
for relation in H_Coh_argument_lengths_clean:
    H_Coh_argument_avg_lengths[relation] = np.mean(H_Coh_argument_lengths_clean[relation])

H_Coh_no_argument_avg_lengths = {}
for relation in H_Coh_no_argument_lengths_clean:
    H_Coh_no_argument_avg_lengths[relation] = np.mean(H_Coh_no_argument_lengths_clean[relation])

### Plot average coherence relation lengths divided by argumentation - Human

# plt.bar(H_Coh_argument_avg_lengths.keys(), H_Coh_argument_avg_lengths.values(), color='b')
# plt.xticks(list(H_Coh_no_argument_counts_clean.keys()),list(H_Coh_no_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=20)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Average length", fontsize=18)
# plt.title('Docs with argument - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(H_Coh_no_argument_avg_lengths.keys(), H_Coh_no_argument_avg_lengths.values(), color='b')
# plt.xticks(list(H_Coh_no_argument_avg_lengths.keys()),list(H_Coh_no_argument_avg_lengths.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=20)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Average length", fontsize=18)
# plt.title('Docs without argument - Human', fontsize=20)
# plt.tight_layout()
# plt.show()

### Clean up and rename coherence relations - argumentation, Grover

G_argument_lengths = {}
G_no_argument_lengths = {}

for annotator in G_Coh_container:
    for number in G_Coh_container[annotator]:
        for relation in G_Coh_container[annotator][number]:
            try:
                tmp = [int(i) for i in relation[0:4]]
                min = np.min(tmp)
                max = np.max(tmp)
                if G_Doc_container[annotator][number][12] != '0': # if argument
                    if relation[4] in G_argument_lengths:
                        G_argument_lengths[relation[4]].append(max-min)
                    else:
                        diff = max-min
                        G_argument_lengths[relation[4]] = [diff]
                else:
                    if relation[4] in G_no_argument_lengths:
                        G_no_argument_lengths[relation[4]].append(max-min)
                    else:
                        diff = max-min
                        G_no_argument_lengths[relation[4]] = [diff]
            except: # because sometimes there's ? instead of a segment number
                pass

G_Coh_argument_lengths_clean = {
        'Cause/effect': [],
        'Elaboration': [],
        'Same': [],
        'Attribution': [],
        'Degenerate': [],
        'Similarity': [],
        'Contrast': [],
        'Temporal sequence': [],
        'Violated expectation': [],
        'Example': [],
        'Condition': [],
        'Generalization': [],
        'Repetition': []
        }

G_Coh_no_argument_lengths_clean = copy.deepcopy(G_Coh_argument_lengths_clean)

clean_up_coh_lengths(G_argument_lengths,G_Coh_argument_lengths_clean)
clean_up_coh_lengths(G_no_argument_lengths,G_Coh_no_argument_lengths_clean)

### Calculate average coherence relation lengths - argumentation, Grover

G_Coh_argument_avg_lengths = {}
for relation in G_Coh_argument_lengths_clean:
    G_Coh_argument_avg_lengths[relation] = np.mean(G_Coh_argument_lengths_clean[relation])

G_Coh_no_argument_avg_lengths = {}
for relation in G_Coh_no_argument_lengths_clean:
    G_Coh_no_argument_avg_lengths[relation] = np.mean(G_Coh_no_argument_lengths_clean[relation])

### Plot average coherence relation lengths divided by argumentation - Grover

# plt.bar(G_Coh_argument_avg_lengths.keys(), G_Coh_argument_avg_lengths.values(), color='g')
# plt.xticks(list(G_Coh_no_argument_counts_clean.keys()),list(G_Coh_no_argument_counts_clean.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=20)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Average length", fontsize=18)
# plt.title('Docs with argument - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(G_Coh_no_argument_avg_lengths.keys(), G_Coh_no_argument_avg_lengths.values(), color='g')
# plt.xticks(list(G_Coh_no_argument_avg_lengths.keys()),list(G_Coh_no_argument_avg_lengths.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=20)
# plt.xlabel("Coherence Relation", fontsize=18)
# plt.ylabel("Average length", fontsize=18)
# plt.title('Docs without argument - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

##############################################################################

### More detailed Situation Entity-Mode of Discourse hypotheses

# HYPOTHESIS: In docs containing narratives, the main referrent of the last
# statement is generic
# HYPOTHESIS: In docs without either narratives or arguments (likely reports),
# more coerced states

### Last generic and coerced report hypotheses - Human

H_last_gen_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}
H_narr_total_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}
H_rep_coerced_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}


for annotator in H_SE_container:
    for number in H_SE_container[annotator]:
        if H_Doc_container[annotator][number][3] != "0" and H_Doc_container[annotator][number][12] != "0":
            H_narr_total_counter['narrative & argument'] += 1
            if "GENERIC" in H_SE_container[annotator][number][len(H_SE_container[annotator][number])-1] or "GNERIC" in H_SE_container[annotator][number][len(H_SE_container[annotator][number])-1]:
                H_last_gen_counter['narrative & argument'] += 1

            for SE in H_SE_container[annotator][number]:
                if "COERCED" in SE:
                    H_rep_coerced_counter['narrative & argument'] += 1

        elif H_Doc_container[annotator][number][3] != "0" and H_Doc_container[annotator][number][12] == "0":
            H_narr_total_counter['narrative'] += 1
            if "GENERIC" in H_SE_container[annotator][number][len(H_SE_container[annotator][number])-1] or "GNERIC" in H_SE_container[annotator][number][len(H_SE_container[annotator][number])-1]:
                H_last_gen_counter['narrative'] += 1

            for SE in H_SE_container[annotator][number]:
                if "COERCED" in SE:
                    H_rep_coerced_counter['narrative'] += 1

        elif H_Doc_container[annotator][number][3] == "0" and H_Doc_container[annotator][number][12] != "0":
            H_narr_total_counter['argument'] += 1
            if "GENERIC" in H_SE_container[annotator][number][len(H_SE_container[annotator][number])-1] or "GNERIC" in H_SE_container[annotator][number][len(H_SE_container[annotator][number])-1]:
                H_last_gen_counter['argument'] += 1

            for SE in H_SE_container[annotator][number]:
                if "COERCED" in SE:
                    H_rep_coerced_counter['argument'] += 1

        elif H_Doc_container[annotator][number][3] == "0" and H_Doc_container[annotator][number][12] == "0":
            H_narr_total_counter['no narrative or argument'] += 1
            if "GENERIC" in H_SE_container[annotator][number][len(H_SE_container[annotator][number])-1] or "GNERIC" in H_SE_container[annotator][number][len(H_SE_container[annotator][number])-1]:
                H_last_gen_counter['no narrative or argument'] += 1

            for SE in H_SE_container[annotator][number]:
                if "COERCED" in SE:
                    H_rep_coerced_counter['no narrative or argument'] += 1

for key in H_last_gen_counter:
    H_last_gen_counter[key] = H_last_gen_counter[key] / H_narr_total_counter[key]

print("***")
print("Human")
print("***")
print("Total count")
print(H_narr_total_counter)
print("Last generic")
print(H_last_gen_counter)
print("Coerced report")
print(H_rep_coerced_counter)

### Last generic and coerced report hypotheses - Grover

G_last_gen_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}
G_narr_total_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}
G_rep_coerced_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}

for annotator in G_SE_container:
    for number in G_SE_container[annotator]:
        if G_Doc_container[annotator][number][3] != "0" and G_Doc_container[annotator][number][12] != "0":
            G_narr_total_counter['narrative & argument'] += 1
            if "GENERIC" in G_SE_container[annotator][number][len(G_SE_container[annotator][number])-1] or "GNERIC" in G_SE_container[annotator][number][len(G_SE_container[annotator][number])-1]:
                G_last_gen_counter['narrative & argument'] += 1

            for SE in G_SE_container[annotator][number]:
                if "COERCED" in SE:
                    G_rep_coerced_counter['narrative & argument'] += 1

        elif G_Doc_container[annotator][number][3] != "0" and G_Doc_container[annotator][number][12] == "0":
            G_narr_total_counter['narrative'] += 1
            if "GENERIC" in G_SE_container[annotator][number][len(G_SE_container[annotator][number])-1] or "GNERIC" in G_SE_container[annotator][number][len(G_SE_container[annotator][number])-1]:
                G_last_gen_counter['narrative'] += 1

            for SE in G_SE_container[annotator][number]:
                if "COERCED" in SE:
                    G_rep_coerced_counter['narrative'] += 1

        elif G_Doc_container[annotator][number][3] == "0" and G_Doc_container[annotator][number][12] != "0":
            G_narr_total_counter['argument'] += 1
            if "GENERIC" in G_SE_container[annotator][number][len(G_SE_container[annotator][number])-1] or "GNERIC" in G_SE_container[annotator][number][len(G_SE_container[annotator][number])-1]:
                G_last_gen_counter['argument'] += 1

            for SE in G_SE_container[annotator][number]:
                if "COERCED" in SE:
                    G_rep_coerced_counter['argument'] += 1

        elif G_Doc_container[annotator][number][3] == "0" and G_Doc_container[annotator][number][12] == "0":
            G_narr_total_counter['no narrative or argument'] += 1
            if "GENERIC" in G_SE_container[annotator][number][len(G_SE_container[annotator][number])-1] or "GNERIC" in G_SE_container[annotator][number][len(G_SE_container[annotator][number])-1]:
                G_last_gen_counter['no narrative or argument'] += 1

            for SE in G_SE_container[annotator][number]:
                if "COERCED" in SE:
                    G_rep_coerced_counter['no narrative or argument'] += 1

for key in G_last_gen_counter:
    G_last_gen_counter[key] = G_last_gen_counter[key] / G_narr_total_counter[key]

print("***")
print("Grover")
print("***")
print("Total count")
print(G_narr_total_counter)
print("Last generic")
print(G_last_gen_counter)
print("Coerced report")
print(G_rep_coerced_counter)

###############################################################################

### Distribution of incoherent and coherent relations - Human

H_incoherent_counter = 0
H_doc_counter = 0
H_incoherent_rels = {'Cause/effect':0,
        'Elaboration':0,
        'Same': 0,
        'Attribution': 0,
        'Degenerate': 0,
        'Similarity': 0,
        'Contrast': 0,
        'Temporal sequence': 0,
        'Violated expectation': 0,
        'Example': 0,
        'Condition': 0,
        'Generalization': 0,
        'Repetition': 0
        }

H_coherent_rels = copy.deepcopy(H_incoherent_rels)

for annotator in H_Coh_container:
    for number in H_Coh_container[annotator]:
        H_doc_counter += 1
        indicator = 0
        for relation in H_Coh_container[annotator][number]:

            if relation[4] == 'cex':
                H_incoherent_rels['Cause/effect'] += 1
                indicator = 1
            elif relation[4] in ['ce', 'cew', 'cer']:
                H_coherent_rels['Cause/effect'] += 1

            elif relation[4] == 'elabx':
                H_incoherent_rels['Elaboration'] += 1
                indicator = 1
            elif relation[4] in ['elab', 'ealb', 'elav', 'elabl']:
                H_coherent_rels['Elaboration'] += 1

            elif relation[4] == 'samex':
                H_incoherent_rels['Same'] += 1
                indicator = 1
            elif relation[4] == 'same':
                H_coherent_rels['Same'] += 1

            elif relation[4] == 'attrx':
                H_incoherent_rels['Attribution'] += 1
                indicator = 1
            elif relation[4] in ['attrm','attr']:
                H_coherent_rels['Attribution'] += 1

            elif relation[4] in ['deg','degenerate','mal']:
                H_incoherent_rels['Degenerate'] += 1
                indicator = 1

            elif relation[4] == 'simx':
                H_incoherent_rels['Similarity'] += 1
                indicator = 1
            elif relation[4] == 'sim':
                H_coherent_rels['Similarity'] += 1

            elif relation[4] == 'contrx':
                H_incoherent_rels['Contrast'] += 1
                indicator = 1
            elif relation[4] == 'contr':
                H_coherent_rels['Contrast'] += 1

            elif relation[4] == 'tempx':
                H_incoherent_rels['Temporal sequence'] += 1
                indicator = 1
            elif relation[4] == 'temp':
                H_coherent_rels['Temporal sequence'] += 1

            elif relation[4] == 'vex':
                H_incoherent_rels['Violated expectation'] += 1
                indicator = 1
            elif relation[4] == 've':
                H_coherent_rels['Violated expectation'] += 1

            elif relation[4] == 'exampx':
                H_incoherent_rels['Example'] += 1
                indicator = 1
            elif relation[4] == 'examp':
                H_coherent_rels['Example'] += 1

            elif relation[4] == 'condx':
                H_incoherent_rels['Condition'] += 1
                indicator = 1
            elif relation[4] == 'cond':
                H_coherent_rels['Condition'] += 1

            elif relation[4] == 'genx':
                H_incoherent_rels['Generalization'] += 1
                indicator = 1
            elif relation[4] == 'gen':
                H_coherent_rels['Generalization'] += 1

            elif relation[4] == 'rep':
                H_incoherent_rels['Repetition'] += 1
                indicator = 1
        if indicator == 1:
            H_incoherent_counter += 1

print("***")
print("Distribution of Incoherent Relations - Human")
print(H_incoherent_rels)
print(H_incoherent_counter / H_doc_counter)

### Distribution of incoherent and coherent relations - Grover

G_incoherent_counter = 0
G_doc_counter = 0
G_incoherent_rels = {'Cause/effect':0,
        'Elaboration':0,
        'Same': 0,
        'Attribution': 0,
        'Degenerate': 0,
        'Similarity': 0,
        'Contrast': 0,
        'Temporal sequence': 0,
        'Violated expectation': 0,
        'Example': 0,
        'Condition': 0,
        'Generalization': 0,
        'Repetition': 0
        }

G_coherent_rels = copy.deepcopy(G_incoherent_rels)

for annotator in G_Coh_container:
    for number in G_Coh_container[annotator]:
        G_doc_counter += 1
        indicator = 0
        for relation in G_Coh_container[annotator][number]:

            if relation[4] == 'cex':
                G_incoherent_rels['Cause/effect'] += 1
                indicator = 1
            elif relation[4] in ['ce', 'cew', 'cer']:
                G_coherent_rels['Cause/effect'] += 1

            elif relation[4] == 'elabx':
                G_incoherent_rels['Elaboration'] += 1
                indicator = 1
            elif relation[4] in ['elab', 'ealb', 'elav', 'elabl']:
                G_coherent_rels['Elaboration'] += 1

            elif relation[4] == 'samex':
                G_incoherent_rels['Same'] += 1
                indicator = 1
            elif relation[4] == 'same':
                G_coherent_rels['Same'] += 1

            elif relation[4] == 'attrx':
                G_incoherent_rels['Attribution'] += 1
                indicator = 1
            elif relation[4] in ['attrm','attr']:
                G_coherent_rels['Attribution'] += 1

            elif relation[4] in ['deg','degenerate','mal']:
                G_incoherent_rels['Degenerate'] += 1
                indicator = 1

            elif relation[4] == 'simx':
                G_incoherent_rels['Similarity'] += 1
                indicator = 1
            elif relation[4] == 'sim':
                G_coherent_rels['Similarity'] += 1

            elif relation[4] == 'contrx':
                G_incoherent_rels['Contrast'] += 1
                indicator = 1
            elif relation[4] == 'contr':
                G_coherent_rels['Contrast'] += 1

            elif relation[4] == 'tempx':
                G_incoherent_rels['Temporal sequence'] += 1
                indicator = 1
            elif relation[4] == 'temp':
                G_coherent_rels['Temporal sequence'] += 1

            elif relation[4] == 'vex':
                G_incoherent_rels['Violated expectation'] += 1
                indicator = 1
            elif relation[4] == 've':
                G_coherent_rels['Violated expectation'] += 1

            elif relation[4] == 'exampx':
                G_incoherent_rels['Example'] += 1
                indicator = 1
            elif relation[4] == 'examp':
                G_coherent_rels['Example'] += 1

            elif relation[4] == 'condx':
                G_incoherent_rels['Condition'] += 1
                indicator = 1
            elif relation[4] == 'cond':
                G_coherent_rels['Condition'] += 1

            elif relation[4] == 'genx':
                G_incoherent_rels['Generalization'] += 1
                indicator = 1
            elif relation[4] == 'gen':
                G_coherent_rels['Generalization'] += 1

            elif relation[4] == 'rep':
                G_incoherent_rels['Repetition'] += 1
                indicator = 1
        if indicator == 1:
            G_incoherent_counter += 1

print("***")
print("Distribution of Incoherent Relations - Human")
print(G_incoherent_rels)
print(G_incoherent_counter / G_doc_counter)

### Frequency plot of incoherent relations - Grover - needed for rep, deg

# plt.bar(G_incoherent_rels.keys(), G_incoherent_rels.values(), color='g')
# plt.xticks(list(G_incoherent_rels.keys()),list(G_incoherent_rels.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=50)
# plt.xlabel("Incoherent Coherence Relation", fontsize=18)
# plt.ylabel("Frequency", fontsize=18)
# plt.title('All docs - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

# G_incoherent_rels.pop('Repetition')
# G_incoherent_rels.pop('Degenerate')
# for key in G_incoherent_rels:
#     G_incoherent_rels[key] = G_incoherent_rels[key] / G_coherent_rels[key]

# ### Proportion plot of incoherent relations - Grover (excluding rep, deg)

# plt.bar(G_incoherent_rels.keys(), G_incoherent_rels.values(), color='g')
# plt.xticks(list(G_incoherent_rels.keys()),list(G_incoherent_rels.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=0.7)
# plt.xlabel("Incoherent Coherence Relation", fontsize=18)
# plt.ylabel("Proportion of relevant relation in texts", fontsize=18)
# plt.title('All docs - Grover', fontsize=20)
# plt.tight_layout()
# plt.show()

## Attitude distribution for Grover and Humans

G_attitudes = {'UNCLEAR':0}
for i in range(6):
    G_attitudes[str(i)] = 0
H_attitudes = {'UNCLEAR':0}
for i in range(6):
    H_attitudes[str(i)] = 0
for annotator in annotators:
    for number in G_Doc_container[annotator]:
        if G_Doc_container[annotator][number][11] in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)']:
            G_attitudes["UNCLEAR"] += 1
        else:
            G_attitudes[G_Doc_container[annotator][number][11]] += 1
    for number in H_Doc_container[annotator]:
        if H_Doc_container[annotator][number][11] in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)']:
            if "UNCLEAR" in H_attitudes:
                H_attitudes["UNCLEAR"] += 1
        else:
            H_attitudes[H_Doc_container[annotator][number][11]] += 1


## Plot Grover and Human distributions side-by-side

labels = ['UNCLEAR']
for i in range(6):
    labels.append(str(i))
G_values = [G_attitudes[i] for i in labels]
H_values = [H_attitudes[i] for i in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x + width/2, G_values, width, label="Grover", color='g')
rects2 = ax.bar(x - width/2, H_values, width, label="Human", color='b')

ax.set_ylabel('Frequency')
ax.set_title('Attitude distributions for Grover and Humans')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

## Make sure the number of doc-level ratings for the document is correct. If not, print out the doc ID

# counter = 0
# for annotator in annotators:
#     for number in H_Doc_container[annotator]:
#         if len(H_Doc_container[annotator][number]) != 16:
#             print(annotator)
#             print(number)
#             counter += 1

# print(counter)

## Correlation between coherence relation frequency and narrativity rating - Human

H_Coh_per_doc = {}

for annotator in annotators:
    if annotator not in H_Coh_per_doc:
        H_Coh_per_doc[annotator] = {}
    for number in H_Coh_container[annotator]:
        if number not in H_Coh_per_doc:
            H_Coh_per_doc[annotator][number] = {}
        for relation in H_Coh_container[annotator][number]:
            if relation[4] in H_Coh_per_doc[annotator][number]:
                H_Coh_per_doc[annotator][number][relation[4]] += 1
            else:
                H_Coh_per_doc[annotator][number][relation[4]] = 1

H_Coh_per_doc_clean = {}

temp = {
        'Cause/effect': 0,
        'Elaboration': 0,
        'Same': 0,
        'Attribution': 0,
        'Degenerate': 0,
        'Similarity': 0,
        'Contrast': 0,
        'Temporal sequence': 0,
        'Violated expectation': 0,
        'Example': 0,
        'Condition': 0,
        'Generalization': 0,
        'Repetition': 0
        }

for annotator in annotators:
    if annotator not in H_Coh_per_doc_clean:
        H_Coh_per_doc_clean[annotator] = {}
    for number in H_Coh_container[annotator]:
        if number not in H_Coh_per_doc_clean:
            H_Coh_per_doc_clean[annotator][number] = {}
        temp1 = copy.deepcopy(temp)
        clean_up_coh_rels(H_Coh_per_doc[annotator][number],temp1)
        H_Coh_per_doc_clean[annotator][number] = temp1


# print("***")
# print("Human coh-narrativity correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     narrativity = []
#     for annotator in annotators:
#         for number in H_Coh_per_doc_clean[annotator]:
#             narrativity.append(int(H_Doc_container[annotator][number][3]))
#             temp2.append(H_Coh_per_doc_clean[annotator][number][relation])
#     print(relation)
#     if abs(pearsonr(narrativity,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(narrativity,temp2))
# print(len(narrativity))

# ## Correlation between coherence relation frequency and narrativity rating - Grover

G_Coh_per_doc = {}

for annotator in annotators:
    if annotator not in G_Coh_per_doc:
        G_Coh_per_doc[annotator] = {}
    for number in G_Coh_container[annotator]:
        if number not in G_Coh_per_doc:
            G_Coh_per_doc[annotator][number] = {}
        for relation in G_Coh_container[annotator][number]:
            if relation[4] in G_Coh_per_doc[annotator][number]:
                G_Coh_per_doc[annotator][number][relation[4]] += 1
            else:
                G_Coh_per_doc[annotator][number][relation[4]] = 1

G_Coh_per_doc_clean = {}

temp = {
        'Cause/effect': 0,
        'Elaboration': 0,
        'Same': 0,
        'Attribution': 0,
        'Degenerate': 0,
        'Similarity': 0,
        'Contrast': 0,
        'Temporal sequence': 0,
        'Violated expectation': 0,
        'Example': 0,
        'Condition': 0,
        'Generalization': 0,
        'Repetition': 0
        }

for annotator in annotators:
    if annotator not in G_Coh_per_doc_clean:
        G_Coh_per_doc_clean[annotator] = {}
    for number in G_Coh_container[annotator]:
        if number not in G_Coh_per_doc_clean:
            G_Coh_per_doc_clean[annotator][number] = {}
        temp1 = copy.deepcopy(temp)
        clean_up_coh_rels(G_Coh_per_doc[annotator][number],temp1)
        G_Coh_per_doc_clean[annotator][number] = temp1

# print("***")
# print("Grover coh-narrativity correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     narrativity = []
#     for annotator in annotators:
#         for number in G_Coh_per_doc_clean[annotator]:
#             narrativity.append(int(G_Doc_container[annotator][number][3]))
#             temp2.append(G_Coh_per_doc_clean[annotator][number][relation])
#     if abs(pearsonr(narrativity,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(narrativity,temp2))
# print(len(narrativity))

# ### Incoherent relations and narrativity - Human

# H_incoh_per_doc = {}

# for annotator in annotators:
#     if annotator not in H_incoh_per_doc:
#         H_incoh_per_doc[annotator] = {}
#     for number in H_Coh_container[annotator]:
#         if number not in H_incoh_per_doc:
#             H_incoh_per_doc[annotator][number] = copy.deepcopy(temp)
#         for relation in H_Coh_container[annotator][number]:
#             if relation[4] == 'cex':
#                 H_incoh_per_doc[annotator][number]['Cause/effect'] += 1
#             elif relation[4] == 'elabx':
#                 H_incoh_per_doc[annotator][number]['Elaboration'] += 1
#             elif relation[4] == 'samex':
#                 H_incoh_per_doc[annotator][number]['Same'] += 1
#             elif relation[4] == 'attrx':
#                 H_incoh_per_doc[annotator][number]['Attribution'] += 1
#             elif relation[4] in ['deg','degenerate','mal']:
#                 H_incoh_per_doc[annotator][number]['Degenerate'] += 1
#             elif relation[4] == 'simx':
#                 H_incoh_per_doc[annotator][number]['Similarity'] += 1
#             elif relation[4] == 'contrx':
#                 H_incoh_per_doc[annotator][number]['Contrast'] += 1
#             elif relation[4] == 'tempx':
#                 H_incoh_per_doc[annotator][number]['Temporal sequence'] += 1
#             elif relation[4] == 'vex':
#                 H_incoh_per_doc[annotator][number]['Violated expectation'] += 1
#             elif relation[4] == 'exampx':
#                 H_incoh_per_doc[annotator][number]['Example'] += 1
#             elif relation[4] == 'condx':
#                 H_incoh_per_doc[annotator][number]['Condition'] += 1
#             elif relation[4] == 'genx':
#                 H_incoh_per_doc[annotator][number]['Generalization'] += 1
#             elif relation[4] == 'rep':
#                 H_incoh_per_doc[annotator][number]['Repetition'] += 1

# print("***")
# print("Human incoh-narrativity correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     narrativity = []
#     for annotator in annotators:
#         for number in H_incoh_per_doc[annotator]:
#             narrativity.append(int(H_Doc_container[annotator][number][3]))
#             temp2.append(H_incoh_per_doc[annotator][number][relation])
#     if abs(pearsonr(narrativity,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(narrativity,temp2))
# print(len(narrativity))

# ### Incoherent relations and narrativity - Grover

# G_incoh_per_doc = {}

# for annotator in annotators:
#     if annotator not in G_incoh_per_doc:
#         G_incoh_per_doc[annotator] = {}
#     for number in G_Coh_container[annotator]:
#         if number not in G_incoh_per_doc:
#             G_incoh_per_doc[annotator][number] = copy.deepcopy(temp)
#         for relation in G_Coh_container[annotator][number]:
#             if relation[4] == 'cex':
#                 G_incoh_per_doc[annotator][number]['Cause/effect'] += 1
#             elif relation[4] == 'elabx':
#                 G_incoh_per_doc[annotator][number]['Elaboration'] += 1
#             elif relation[4] == 'samex':
#                 G_incoh_per_doc[annotator][number]['Same'] += 1
#             elif relation[4] == 'attrx':
#                 G_incoh_per_doc[annotator][number]['Attribution'] += 1
#             elif relation[4] in ['deg','degenerate','mal']:
#                 G_incoh_per_doc[annotator][number]['Degenerate'] += 1
#             elif relation[4] == 'simx':
#                 G_incoh_per_doc[annotator][number]['Similarity'] += 1
#             elif relation[4] == 'contrx':
#                 G_incoh_per_doc[annotator][number]['Contrast'] += 1
#             elif relation[4] == 'tempx':
#                 G_incoh_per_doc[annotator][number]['Temporal sequence'] += 1
#             elif relation[4] == 'vex':
#                 G_incoh_per_doc[annotator][number]['Violated expectation'] += 1
#             elif relation[4] == 'exampx':
#                 G_incoh_per_doc[annotator][number]['Example'] += 1
#             elif relation[4] == 'condx':
#                 G_incoh_per_doc[annotator][number]['Condition'] += 1
#             elif relation[4] == 'genx':
#                 G_incoh_per_doc[annotator][number]['Generalization'] += 1
#             elif relation[4] == 'rep':
#                 G_incoh_per_doc[annotator][number]['Repetition'] += 1

# print("***")
# print("Grover incoh-narrativity correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     narrativity = []
#     for annotator in annotators:
#         for number in G_incoh_per_doc[annotator]:
#             narrativity.append(int(G_Doc_container[annotator][number][3]))
#             temp2.append(G_incoh_per_doc[annotator][number][relation])
#     if abs(pearsonr(narrativity,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(narrativity,temp2))
# print(len(narrativity))

# ## Correlation between coherence relation frequency and attitude rating - Human

# print("***")
# print("Human coh-attitude correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     attitude = []
#     for annotator in annotators:
#         for number in H_Coh_per_doc_clean[annotator]:
#             if H_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)'] and H_Doc_container[annotator][number][11] != '0':
#                 attitude.append(int(H_Doc_container[annotator][number][11]))
#                 temp2.append(H_Coh_per_doc_clean[annotator][number][relation])
#     if abs(pearsonr(attitude,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(attitude,temp2))
# print(len(attitude))

# ## Correlation between coherence relation frequency and attitude rating - Grover

# print("***")
# print("Grover coh-attitude correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     attitude = []
#     for annotator in annotators:
#         for number in G_Coh_per_doc_clean[annotator]:
#             if G_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)'] and G_Doc_container[annotator][number][11] != '0':
#                 attitude.append(int(G_Doc_container[annotator][number][11]))
#                 temp2.append(G_Coh_per_doc_clean[annotator][number][relation])
#     if abs(pearsonr(attitude,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(attitude,temp2))
# print(len(attitude))

# ## Correlation between incoherence frequency and attitude rating - Human

# print("***")
# print("Human incoh-attitude correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     attitude = []
#     for annotator in annotators:
#         for number in H_incoh_per_doc[annotator]:
#             if H_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)'] and H_Doc_container[annotator][number][11] != '0':
#                 attitude.append(int(H_Doc_container[annotator][number][11]))
#                 temp2.append(H_incoh_per_doc[annotator][number][relation])
#     if abs(pearsonr(attitude,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(attitude,temp2))
# print(len(attitude))

# ## Correlation between incoherence frequency and attitude rating - Grover

# print("***")
# print("Grover incoh-attitude correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     attitude = []
#     for annotator in annotators:
#         for number in G_incoh_per_doc[annotator]:
#             if G_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)'] and G_Doc_container[annotator][number][11] != '0':
#                 attitude.append(int(G_Doc_container[annotator][number][11]))
#                 temp2.append(G_incoh_per_doc[annotator][number][relation])
#     if abs(pearsonr(attitude,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(attitude,temp2))
# print(len(attitude))

# ## Correlation between coherence relation frequency and narrative quality ratings - Human

# print("***")
# print("Human coherence-narrative quality correlations")
# print("***")

# for dimension in range(4):
#     print("Dimension {}".format(dimension+1))
#     for relation in temp:
#         temp2 = []
#         dim = []
#         for annotator in annotators:
#             for number in H_Coh_per_doc_clean[annotator]:
#                 if H_Doc_container[annotator][number][3] != '0':
#                     dim.append(int(H_Doc_container[annotator][number][4+dimension]))
#                     temp2.append(H_Coh_per_doc_clean[annotator][number][relation])
#         if abs(pearsonr(dim,temp2)[0] > corr_threshold):
#             print(relation)
#             print(pearsonr(dim,temp2))
#     print(len(dim))

# ## Correlation between coherence relation frequency and narrative quality ratings - Grover

# print("***")
# print("Grover coherence-narrative quality correlations")
# print("***")

# for dimension in range(4):
#     print("Dimension {}".format(dimension+1))
#     for relation in temp:
#         temp2 = []
#         dim = []
#         for annotator in annotators:
#             for number in G_Coh_per_doc_clean[annotator]:
#                 if G_Doc_container[annotator][number][3] != '0':
#                     dim.append(int(G_Doc_container[annotator][number][4+dimension]))
#                     temp2.append(G_Coh_per_doc_clean[annotator][number][relation])
#         if abs(pearsonr(dim,temp2)[0] > corr_threshold):
#             print(relation)
#             print(pearsonr(dim,temp2))
#     print(len(dim))

# ## Correlation between incoherence frequency and narrative quality ratings - Human

# print("***")
# print("Human incoherence-narrative quality correlations")
# print("***")

# for dimension in range(4):
#     print("Dimension {}".format(dimension+1))
#     for relation in temp:
#         temp2 = []
#         dim = []
#         for annotator in annotators:
#             for number in H_incoh_per_doc[annotator]:
#                 if H_Doc_container[annotator][number][3] != '0':
#                     dim.append(int(H_Doc_container[annotator][number][4+dimension]))
#                     temp2.append(H_incoh_per_doc[annotator][number][relation])
#         if abs(pearsonr(dim,temp2)[0] > corr_threshold):
#             print(relation)
#             print(pearsonr(dim,temp2))
#     print(len(dim))

# ## Correlation between incoherence relation frequency and narrative quality ratings - Grover

# print("***")
# print("Grover incoherence-narrative quality correlations")
# print("***")

# for dimension in range(4):
#     print("Dimension {}".format(dimension+1))
#     for relation in temp:
#         temp2 = []
#         dim = []
#         for annotator in annotators:
#             for number in G_incoh_per_doc[annotator]:
#                 if G_Doc_container[annotator][number][3] != '0':
#                     dim.append(int(G_Doc_container[annotator][number][4+dimension]))
#                     temp2.append(G_incoh_per_doc[annotator][number][relation])
#         if abs(pearsonr(dim,temp2)[0] > corr_threshold):
#             print(relation)
#             print(pearsonr(dim,temp2))
#     print(len(dim))

## Correlation between coherence frequency and argumentation - Human

# print("***")
# print("Human coh-argumentation correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     argument = []
#     for annotator in annotators:
#         for number in H_Coh_per_doc_clean[annotator]:
#             if H_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)'] and H_Doc_container[annotator][number][11] != '0':
#                 argument.append(1)
#                 temp2.append(H_Coh_per_doc_clean[annotator][number][relation])
#             else:
#                 argument.append(0)
#                 temp2.append(H_Coh_per_doc_clean[annotator][number][relation])
#     if abs(pearsonr(argument,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(argument,temp2))
# print(len(argument))

# ## Correlation between coherence frequency and argumentation - Grover

# print("***")
# print("Grover coh-argumentation correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     argument = []
#     for annotator in annotators:
#         for number in G_Coh_per_doc_clean[annotator]:
#             if G_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)'] and G_Doc_container[annotator][number][11] != '0':
#                 argument.append(1)
#                 temp2.append(G_Coh_per_doc_clean[annotator][number][relation])
#             else:
#                 argument.append(0)
#                 temp2.append(G_Coh_per_doc_clean[annotator][number][relation])
#     if abs(pearsonr(argument,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(argument,temp2))
# print(len(argument))

# ## Correlation between incoherence frequency and argumentation - Human

# print("***")
# print("Human incoh-argumentation correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     argument = []
#     for annotator in annotators:
#         for number in H_incoh_per_doc[annotator]:
#             if H_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)'] and H_Doc_container[annotator][number][11] != '0':
#                 argument.append(1)
#                 temp2.append(H_incoh_per_doc[annotator][number][relation])
#             else:
#                 argument.append(0)
#                 temp2.append(H_incoh_per_doc[annotator][number][relation])
#     if abs(pearsonr(argument,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(argument,temp2))
# print(len(argument))

# ## Correlation between incoherence frequency and argumentation - Grover

# print("***")
# print("Grover incoh-argumentation correlations")
# print("***")

# for relation in temp:
#     temp2 = []
#     argument = []
#     for annotator in annotators:
#         for number in G_incoh_per_doc[annotator]:
#             if G_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)'] and G_Doc_container[annotator][number][11] != '0':
#                 argument.append(1)
#                 temp2.append(G_incoh_per_doc[annotator][number][relation])
#             else:
#                 argument.append(0)
#                 temp2.append(G_incoh_per_doc[annotator][number][relation])
#     if abs(pearsonr(argument,temp2)[0] > corr_threshold):
#         print(relation)
#         print(pearsonr(argument,temp2))
# print(len(argument))

# ## Correlation between coherence frequency and argument quality ratings - Human

# print("***")
# print("Human coherence-argument quality correlations")
# print("***")

# for dimension in range(3):
#     print("Dimension {}".format(dimension+1))
#     for relation in temp:
#         temp2 = []
#         dim = []
#         for annotator in annotators:
#             for number in H_Coh_per_doc_clean[annotator]:
#                 if H_Doc_container[annotator][number][11] != '0' and H_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)']:
#                     dim.append(int(H_Doc_container[annotator][number][12+dimension]))
#                     temp2.append(H_Coh_per_doc_clean[annotator][number][relation])
#         if abs(pearsonr(dim,temp2)[0] > corr_threshold):
#             print(relation)
#             print(pearsonr(dim,temp2))
#     print(len(dim))

# ## Correlation between coherence frequency and argument quality ratings - Grover

# print("***")
# print("Grover coherence-argument quality correlations")
# print("***")

# for dimension in range(3):
#     print("Dimension {}".format(dimension+1))
#     for relation in temp:
#         temp2 = []
#         dim = []
#         for annotator in annotators:
#             for number in G_Coh_per_doc_clean[annotator]:
#                 if G_Doc_container[annotator][number][11] != '0' and G_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)']:
#                     dim.append(int(G_Doc_container[annotator][number][12+dimension]))
#                     temp2.append(G_Coh_per_doc_clean[annotator][number][relation])
#         if abs(pearsonr(dim,temp2)[0] > corr_threshold):
#             print(relation)
#             print(pearsonr(dim,temp2))
#     print(len(dim))

# ## Correlation between incoherence frequency and argument quality ratings - Human

# print("***")
# print("Human incoherence-argument quality correlations")
# print("***")

# for dimension in range(3):
#     print("Dimension {}".format(dimension+1))
#     for relation in temp:
#         temp2 = []
#         dim = []
#         for annotator in annotators:
#             for number in H_incoh_per_doc[annotator]:
#                 if H_Doc_container[annotator][number][11] != '0' and H_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)']:
#                     dim.append(int(H_Doc_container[annotator][number][12+dimension]))
#                     temp2.append(H_incoh_per_doc[annotator][number][relation])
#         if abs(pearsonr(dim,temp2)[0] > corr_threshold):
#             print(relation)
#             print(pearsonr(dim,temp2))
#     print(len(dim))

# ## Correlation between incoherence frequency and argument quality ratings - Grover

# print("***")
# print("Grover incoherence-argument quality correlations")
# print("***")

# for dimension in range(3):
#     print("Dimension {}".format(dimension+1))
#     for relation in temp:
#         temp2 = []
#         dim = []
#         for annotator in annotators:
#             for number in G_incoh_per_doc[annotator]:
#                 if G_Doc_container[annotator][number][11] != '0' and G_Doc_container[annotator][number][11] not in ['UNCLEAR','UNCLEAR(2)','UNCLEAR(4)']:
#                     dim.append(int(G_Doc_container[annotator][number][12+dimension]))
#                     temp2.append(G_incoh_per_doc[annotator][number][relation])
#         if abs(pearsonr(dim,temp2)[0] > corr_threshold):
#             print(relation)
#             print(pearsonr(dim,temp2))
#     print(len(dim))

### Mean and standard error of the number of coherence relations in a document, comparing Human and Grover

H_relation_count = []

for annotator in annotators:
    for number in H_Coh_per_doc_clean[annotator]:
        H_relation_count.append(np.sum(list(H_Coh_per_doc_clean[annotator][number].values())))

print("***")
print("Mean and standard error of the number of coherence relations in a document - Human")
print("***")
print(np.mean(H_relation_count))
print(scipy.stats.sem(H_relation_count))

G_relation_count = []

for annotator in annotators:
    for number in G_Coh_per_doc_clean[annotator]:
        G_relation_count.append(np.sum(list(G_Coh_per_doc_clean[annotator][number].values())))

print("***")
print("Mean and standard error of the number of coherence relations in a document - Grover")
print("***")
print(np.mean(G_relation_count))
print(scipy.stats.sem(G_relation_count))

### Average number of particular kinds of coherence relations in docs with arguments for human/Grover

### Human

H_Coh_no_narrative_per_doc_counts = {}
H_Coh_narrative_per_doc_counts = {}
H_Coh_no_argument_per_doc_counts = {}
H_Coh_argument_per_doc_counts = {}

for annotator in H_Coh_container:

    for number in H_Coh_container[annotator]:
        ratings = H_Doc_container[annotator][number]

        if ratings[3] != '0': # narrative

            if annotator not in H_Coh_narrative_per_doc_counts:
                H_Coh_narrative_per_doc_counts[annotator] = {}
            if number not in H_Coh_narrative_per_doc_counts[annotator]:
                H_Coh_narrative_per_doc_counts[annotator][number] = {}

            for relation in H_Coh_per_doc_clean[annotator][number]:
                H_Coh_narrative_per_doc_counts[annotator][number][relation] = H_Coh_per_doc_clean[annotator][number][relation]

        else: # if not a narrative
            
            if annotator not in H_Coh_no_narrative_per_doc_counts:
                H_Coh_no_narrative_per_doc_counts[annotator] = {}
            if number not in H_Coh_no_narrative_per_doc_counts[annotator]:
                H_Coh_no_narrative_per_doc_counts[annotator][number] = {}

            for relation in H_Coh_per_doc_clean[annotator][number]:
                H_Coh_no_narrative_per_doc_counts[annotator][number][relation] = H_Coh_per_doc_clean[annotator][number][relation]

        if ratings[12] != '0': # argument
            
            if annotator not in H_Coh_argument_per_doc_counts:
                H_Coh_argument_per_doc_counts[annotator] = {}
            if number not in H_Coh_argument_per_doc_counts[annotator]:
                H_Coh_argument_per_doc_counts[annotator][number] = {}

            for relation in H_Coh_per_doc_clean[annotator][number]:
                H_Coh_argument_per_doc_counts[annotator][number][relation] = H_Coh_per_doc_clean[annotator][number][relation]

        else:
            
            if annotator not in H_Coh_no_argument_per_doc_counts:
                H_Coh_no_argument_per_doc_counts[annotator] = {}
            if number not in H_Coh_no_argument_per_doc_counts[annotator]:
                H_Coh_no_argument_per_doc_counts[annotator][number] = {}

            for relation in H_Coh_per_doc_clean[annotator][number]:
                H_Coh_no_argument_per_doc_counts[annotator][number][relation] = H_Coh_per_doc_clean[annotator][number][relation]

H_Coh_no_narrative_per_doc_proportion = {relation:[] for relation in H_Coh_no_argument_per_doc_counts[annotator][number]}
for annotator in H_Coh_no_narrative_per_doc_counts:
    for number in H_Coh_no_narrative_per_doc_counts[annotator]:
        for relation in H_Coh_no_narrative_per_doc_counts[annotator][number]:
            H_Coh_no_narrative_per_doc_proportion[relation].append(H_Coh_no_narrative_per_doc_counts[annotator][number][relation] / np.sum(list(H_Coh_no_narrative_per_doc_counts[annotator][number].values())))

H_Coh_no_narrative_per_doc_average = {}
for relation in H_Coh_no_argument_per_doc_counts[annotator][number]:
    H_Coh_no_narrative_per_doc_average[relation] = np.mean(list(H_Coh_no_narrative_per_doc_proportion[relation]))

H_Coh_narrative_per_doc_proportion = {relation:[] for relation in H_Coh_no_argument_per_doc_counts[annotator][number]}
for annotator in H_Coh_narrative_per_doc_counts:
    for number in H_Coh_narrative_per_doc_counts[annotator]:
        for relation in H_Coh_narrative_per_doc_counts[annotator][number]:
            H_Coh_narrative_per_doc_proportion[relation].append(H_Coh_narrative_per_doc_counts[annotator][number][relation] / np.sum(list(H_Coh_narrative_per_doc_counts[annotator][number].values())))

H_Coh_narrative_per_doc_average = {}
for relation in H_Coh_no_argument_per_doc_counts[annotator][number]:
    H_Coh_narrative_per_doc_average[relation] = np.mean(list(H_Coh_narrative_per_doc_proportion[relation]))

H_Coh_no_argument_per_doc_proportion = {relation:[] for relation in H_Coh_no_argument_per_doc_counts[annotator][number]}
for annotator in H_Coh_no_argument_per_doc_counts:
    for number in H_Coh_no_argument_per_doc_counts[annotator]:
        for relation in H_Coh_no_argument_per_doc_counts[annotator][number]:
            H_Coh_no_argument_per_doc_proportion[relation].append(H_Coh_no_argument_per_doc_counts[annotator][number][relation] / np.sum(list(H_Coh_no_argument_per_doc_counts[annotator][number].values())))

H_Coh_no_argument_per_doc_average = {}
for relation in H_Coh_no_argument_per_doc_counts[annotator][number]:
    H_Coh_no_argument_per_doc_average[relation] = np.mean(list(H_Coh_no_argument_per_doc_proportion[relation]))

H_Coh_argument_per_doc_proportion = {relation:[] for relation in H_Coh_no_argument_per_doc_counts[annotator][number]}
for annotator in H_Coh_argument_per_doc_counts:
    for number in H_Coh_argument_per_doc_counts[annotator]:
        for relation in H_Coh_argument_per_doc_counts[annotator][number]:
            H_Coh_argument_per_doc_proportion[relation].append(H_Coh_argument_per_doc_counts[annotator][number][relation] / np.sum(list(H_Coh_argument_per_doc_counts[annotator][number].values())))

H_Coh_argument_per_doc_average = {}
for relation in H_Coh_argument_per_doc_counts[annotator][number]:
    H_Coh_argument_per_doc_average[relation] = np.mean(list(H_Coh_argument_per_doc_proportion[relation]))

### Grover

G_Coh_no_narrative_per_doc_counts = {}
G_Coh_narrative_per_doc_counts = {}
G_Coh_no_argument_per_doc_counts = {}
G_Coh_argument_per_doc_counts = {}

for annotator in G_Coh_container:

    for number in G_Coh_container[annotator]:
        ratings = G_Doc_container[annotator][number]

        if ratings[3] != '0': # narrative

            if annotator not in G_Coh_narrative_per_doc_counts:
                G_Coh_narrative_per_doc_counts[annotator] = {}
            if number not in G_Coh_narrative_per_doc_counts[annotator]:
                G_Coh_narrative_per_doc_counts[annotator][number] = {}

            for relation in G_Coh_per_doc_clean[annotator][number]:
                G_Coh_narrative_per_doc_counts[annotator][number][relation] = G_Coh_per_doc_clean[annotator][number][relation]

        else: # if not a narrative
            
            if annotator not in G_Coh_no_narrative_per_doc_counts:
                G_Coh_no_narrative_per_doc_counts[annotator] = {}
            if number not in G_Coh_no_narrative_per_doc_counts[annotator]:
                G_Coh_no_narrative_per_doc_counts[annotator][number] = {}

            for relation in G_Coh_per_doc_clean[annotator][number]:
                G_Coh_no_narrative_per_doc_counts[annotator][number][relation] = G_Coh_per_doc_clean[annotator][number][relation]

        if ratings[12] != '0': # argument
            
            if annotator not in G_Coh_argument_per_doc_counts:
                G_Coh_argument_per_doc_counts[annotator] = {}
            if number not in G_Coh_argument_per_doc_counts[annotator]:
                G_Coh_argument_per_doc_counts[annotator][number] = {}

            for relation in G_Coh_per_doc_clean[annotator][number]:
                G_Coh_argument_per_doc_counts[annotator][number][relation] = G_Coh_per_doc_clean[annotator][number][relation]

        else:
            
            if annotator not in G_Coh_no_argument_per_doc_counts:
                G_Coh_no_argument_per_doc_counts[annotator] = {}
            if number not in G_Coh_no_argument_per_doc_counts[annotator]:
                G_Coh_no_argument_per_doc_counts[annotator][number] = {}

            for relation in G_Coh_per_doc_clean[annotator][number]:
                G_Coh_no_argument_per_doc_counts[annotator][number][relation] = G_Coh_per_doc_clean[annotator][number][relation]

G_Coh_no_narrative_per_doc_proportion = {relation:[] for relation in G_Coh_no_argument_per_doc_counts[annotator][number]}
for annotator in G_Coh_no_narrative_per_doc_counts:
    for number in G_Coh_no_narrative_per_doc_counts[annotator]:
        for relation in G_Coh_no_narrative_per_doc_counts[annotator][number]:
            G_Coh_no_narrative_per_doc_proportion[relation].append(G_Coh_no_narrative_per_doc_counts[annotator][number][relation] / np.sum(list(G_Coh_no_narrative_per_doc_counts[annotator][number].values())))

G_Coh_no_narrative_per_doc_average = {}
for relation in G_Coh_no_argument_per_doc_counts[annotator][number]:
    G_Coh_no_narrative_per_doc_average[relation] = np.mean(list(G_Coh_no_narrative_per_doc_proportion[relation]))

G_Coh_narrative_per_doc_proportion = {relation:[] for relation in G_Coh_no_argument_per_doc_counts[annotator][number]}
for annotator in G_Coh_narrative_per_doc_counts:
    for number in G_Coh_narrative_per_doc_counts[annotator]:
        for relation in G_Coh_narrative_per_doc_counts[annotator][number]:
            G_Coh_narrative_per_doc_proportion[relation].append(G_Coh_narrative_per_doc_counts[annotator][number][relation] / np.sum(list(G_Coh_narrative_per_doc_counts[annotator][number].values())))

G_Coh_narrative_per_doc_average = {}
for relation in G_Coh_no_argument_per_doc_counts[annotator][number]:
    G_Coh_narrative_per_doc_average[relation] = np.mean(list(G_Coh_narrative_per_doc_proportion[relation]))

G_Coh_no_argument_per_doc_proportion = {relation:[] for relation in G_Coh_no_argument_per_doc_counts[annotator][number]}
for annotator in G_Coh_no_argument_per_doc_counts:
    for number in G_Coh_no_argument_per_doc_counts[annotator]:
        for relation in G_Coh_no_argument_per_doc_counts[annotator][number]:
            G_Coh_no_argument_per_doc_proportion[relation].append(G_Coh_no_argument_per_doc_counts[annotator][number][relation] / np.sum(list(G_Coh_no_argument_per_doc_counts[annotator][number].values())))

G_Coh_no_argument_per_doc_average = {}
for relation in G_Coh_no_argument_per_doc_counts[annotator][number]:
    G_Coh_no_argument_per_doc_average[relation] = np.mean(list(G_Coh_no_argument_per_doc_proportion[relation]))

G_Coh_argument_per_doc_proportion = {relation:[] for relation in G_Coh_no_argument_per_doc_counts[annotator][number]}
for annotator in G_Coh_argument_per_doc_counts:
    for number in G_Coh_argument_per_doc_counts[annotator]:
        for relation in G_Coh_argument_per_doc_counts[annotator][number]:
            G_Coh_argument_per_doc_proportion[relation].append(G_Coh_argument_per_doc_counts[annotator][number][relation] / np.sum(list(G_Coh_argument_per_doc_counts[annotator][number].values())))

G_Coh_argument_per_doc_average = {}
for relation in G_Coh_argument_per_doc_counts[annotator][number]:
    G_Coh_argument_per_doc_average[relation] = np.mean(list(G_Coh_argument_per_doc_proportion[relation]))

### Comparative plots of Humans-Grover for the same document type - Narrative

# No narrative

labels = list(G_Coh_argument_per_doc_counts[annotator][number].keys())
G_values = [G_Coh_no_narrative_per_doc_average[i] for i in labels]
H_values = [H_Coh_no_narrative_per_doc_average[i] for i in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x + width/2, G_values, width, label="Grover", color='g')
rects2 = ax.bar(x - width/2, H_values, width, label="Human", color='b')

ax.set_ylabel('Within-document proportion of \ncoherence relations')
ax.set_title('For Grover and Humans - Docs containing no narratives')
ax.set_xticks(x)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
ax.set_ylim(0,0.15)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

# Narrative

labels = list(G_Coh_argument_per_doc_counts[annotator][number].keys())
G_values = [G_Coh_narrative_per_doc_average[i] for i in labels]
H_values = [H_Coh_narrative_per_doc_average[i] for i in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x + width/2, G_values, width, label="Grover", color='g')
rects2 = ax.bar(x - width/2, H_values, width, label="Human", color='b')

ax.set_ylabel('Within-document proportion of \ncoherence relations')
ax.set_title('For Grover and Humans - Docs containing narratives')
ax.set_xticks(x)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
ax.set_ylim(0,0.15)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

### Comparative plots of Humans-Grover for the same document type - Argument

# No argument

labels = list(G_Coh_argument_per_doc_counts[annotator][number].keys())
G_values = [G_Coh_no_argument_per_doc_average[i] for i in labels]
H_values = [H_Coh_no_argument_per_doc_average[i] for i in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x + width/2, G_values, width, label="Grover", color='g')
rects2 = ax.bar(x - width/2, H_values, width, label="Human", color='b')

ax.set_ylabel('Within-document proportion of \ncoherence relations')
ax.set_title('For Grover and Humans - Docs containing no arguments')
ax.set_xticks(x)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
ax.set_ylim(0,0.15)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

# Narrative

labels = list(G_Coh_argument_per_doc_counts[annotator][number].keys())
G_values = [G_Coh_argument_per_doc_average[i] for i in labels]
H_values = [H_Coh_argument_per_doc_average[i] for i in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x + width/2, G_values, width, label="Grover", color='g')
rects2 = ax.bar(x - width/2, H_values, width, label="Human", color='b')

ax.set_ylabel('Within-document proportion of \ncoherence relations')
ax.set_title('For Grover and Humans - Docs containing arguments')
ax.set_xticks(x)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
ax.set_ylim(0,0.15)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()