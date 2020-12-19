# -*- coding: utf-8 -*-

# TODO: adjudicate between annotators. Currently we're taking only one for each
# shared document

# Counts updated on 12/10/2020

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
    print("  " + element)
    print("  " + str(len(H_SE_container[element])))

print("Grover")
for element in G_SE_container:
    print("  " + element)
    print("  " + str(len(G_SE_container[element])))

print("***")
print("Coherence")
print("***")

print("Human")
for element in H_Coh_container:
    print("  " + element)
    print("  " + str(len(H_Coh_container[element])))

print("Grover")
for element in G_Coh_container:
    print("  " + element)
    print("  " + str(len(G_Coh_container[element])))

################################################################################

### Narrativity distribution for Human and Grover documents

### function that gets counts for SE types for all narrative/non-narrative docs 
# gets counts for SE types for all narrative docs
# weigh by amount that is narrative
def narrative_SE_counts(Doc_container, SE_container, no_narrative_counts, narrative_counts, narrativity_counts):
    for annotator in SE_container.keys():
        for doc_id in SE_container[annotator].keys():
            ratings = Doc_container[annotator][doc_id]
            SE_list = SE_container[annotator][doc_id]

            narrativity_counts[int(ratings[3].strip())] += 1

            if ratings[3] == '0': # if no narrative
                for SE in SE_list:
                    if SE.strip() in no_narrative_counts.keys():
                        no_narrative_counts[SE.strip()] += 1
                    else:
                        no_narrative_counts[SE.strip()] = 1
            elif ratings[3] in ['1', '2', '3', '4', '5']: # if there is narrative
                for SE in SE_list:
                    if SE.strip() in narrative_counts.keys():
                        narrative_counts[SE.strip()] += 1 * (int(ratings[3]) / 5)
                    else:
                        narrative_counts[SE.strip()] = 1 * (int(ratings[3]) / 5)
    
### use function to calculate narrativity distribution for human documents
H_no_narrative_counts = {}
H_narrative_counts = {}
H_narrativity_counts = {i:0 for i in range(6)}
narrative_SE_counts(H_Doc_container, H_SE_container, H_no_narrative_counts, H_narrative_counts, H_narrativity_counts)

### same for grover documents
G_no_narrative_counts = {}
G_narrative_counts = {}
G_narrativity_counts = {i:0 for i in range(6)}
narrative_SE_counts(G_Doc_container, G_SE_container, G_no_narrative_counts, G_narrative_counts, G_narrativity_counts)


### print the narrativity distributions for human and grover
print("***")
print("Human Narrativity distribution")
print("***")
print(H_narrativity_counts)

print("***")
print("Grover Narrativity distribution")
print("***")
print(G_narrativity_counts)

### plot the narrativity distributions for human and grover

# plt.bar(H_narrativity_counts.keys(), H_narrativity_counts.values(), color='b')
# plt.xticks(list(H_narrativity_counts.keys()),list(H_narrativity_counts.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=35)
# plt.xlabel("Proportion narrative in a document \n(0: none; 5: all)",fontsize=18)
# plt.ylabel("Frequency",fontsize=18)
# plt.title('Human-generated documents',fontsize=20)
# plt.tight_layout()
# plt.show()

# plt.bar(G_narrativity_counts.keys(), G_narrativity_counts.values(), color='g')
# plt.xticks(list(G_narrativity_counts.keys()),list(G_narrativity_counts.keys()), rotation='vertical',fontsize=14)
# plt.axis(ymax=35)
# plt.xlabel("Proportion narrative in a document \n(0: none; 5: all)",fontsize=18)
# plt.ylabel("Frequency",fontsize=18)
# plt.title('Grover-generated documents',fontsize=20)
# plt.tight_layout()
# plt.show()


################################################################################

### Clean up and rename Situation Entity annotations - Human and Grover

# HYPOTHESIS: narrative documents should have higher avg bounded events and states
# average up for bounded events and states for both dictionaries

### function that takes an old_dict and a clean_dict and loads old into clean
### clean_dict should have keys defined already for each SE type
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

### define clean dictionaries for human and grover, narrative and no narrative
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
G_no_narrative_counts_clean = copy.deepcopy(H_no_narrative_counts_clean)
G_narrative_counts_clean = copy.deepcopy(H_no_narrative_counts_clean)

### clean up human no narrative and narrative SE counts
clean_up_SE_types(H_no_narrative_counts, H_no_narrative_counts_clean)
clean_up_SE_types(H_narrative_counts, H_narrative_counts_clean)

### clean up grover no narrative and narrative SE counts
clean_up_SE_types(G_no_narrative_counts, G_no_narrative_counts_clean)
clean_up_SE_types(G_narrative_counts, G_narrative_counts_clean)


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

### function that gets counts for SE types for all argument/non-argument docs 
def argument_SE_counts(Doc_container, SE_container, no_argument_counts, argument_counts, argumentation):
    for annotator in SE_container.keys():
        for number in SE_container[annotator].keys():

            ratings = Doc_container[annotator][number]
            SE_list = SE_container[annotator][number]

            if ratings[12] == '0': # if no argument

                argumentation["no argument"] += 1

                for SE in SE_list:
                    if SE.strip() in no_argument_counts.keys():
                        no_argument_counts[SE.strip()] += 1
                    else:
                        no_argument_counts[SE.strip()] = 1
            elif ratings[12] in ['1', '2', '3']: # if there is argument

                argumentation["argument"] += 1

                for SE in SE_list:
                    if SE.strip() in argument_counts.keys():
                        argument_counts[SE.strip()] += 1
                    else:
                        argument_counts[SE.strip()] = 1

### get SE counts for arguments for human-generated documents
H_no_argument_counts = {}
H_argument_counts = {}
H_argumentation = {"argument":0,"no argument":0}
argument_SE_counts(H_Doc_container, H_SE_container, H_no_argument_counts, H_argument_counts, H_argumentation)

### get SE counts for arguments for grover-generated documents
G_no_argument_counts = {}
G_argument_counts = {}
G_argumentation = {"argument":0,"no argument":0}
argument_SE_counts(G_Doc_container, G_SE_container, G_no_argument_counts, G_argument_counts, G_argumentation)



### Clean up and rename Situation Entity annotations - Human and Grover
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
G_no_argument_counts_clean = copy.deepcopy(H_no_argument_counts_clean)
G_argument_counts_clean = copy.deepcopy(H_no_argument_counts_clean)

### clean up for both dicts for human and grover
clean_up_SE_types(H_no_argument_counts, H_no_argument_counts_clean)
clean_up_SE_types(H_argument_counts, H_argument_counts_clean)
clean_up_SE_types(G_no_argument_counts, G_no_argument_counts_clean)
clean_up_SE_types(G_argument_counts, G_argument_counts_clean)

### print distributions for human and grover for arguments
print("***")
print("Human argumentation distribution")
print("***")
print(H_argumentation)

print("***")
print("Grover argumentation distribution")
print("***")
print(G_argumentation)

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
### Extract coherence relations from annotated documents divided by narrativity and argumentation

### function that for each document, loads coherence counts into given dictionaries
def coh_counts(Coh_container, Doc_container, Coh_narrative_counts, Coh_no_narrative_counts, Coh_argument_counts, Coh_no_argument_counts):
    for annotator in Coh_container:
        for number in Coh_container[annotator]:
            
            if number == 51020131204: # special cased grover doc
                del Coh_container[annotator][number][0]

            ratings = Doc_container[annotator][number]

            if ratings[3] != '0': # narrative
                for element in Coh_container[annotator][number]:
                    if element[4] not in Coh_narrative_counts:
                        Coh_narrative_counts[element[4]] = 1
                    else:
                        Coh_narrative_counts[element[4]] += 1
            else:
                for element in Coh_container[annotator][number]:
                    if element[4] not in Coh_no_narrative_counts:
                        Coh_no_narrative_counts[element[4]] = 1
                    else:
                        Coh_no_narrative_counts[element[4]] += 1

            if ratings[12] != '0': # argument
                for element in Coh_container[annotator][number]:
                    if element[4] not in Coh_argument_counts:
                        Coh_argument_counts[element[4]] = 1
                    else:
                        Coh_argument_counts[element[4]] += 1
            else:
                for element in Coh_container[annotator][number]:
                    if element[4] not in Coh_no_argument_counts:
                        Coh_no_argument_counts[element[4]] = 1
                    else:
                        Coh_no_argument_counts[element[4]] += 1

### Human coherence counts
H_Coh_no_narrative_counts = {}
H_Coh_narrative_counts = {}
H_Coh_no_argument_counts = {}
H_Coh_argument_counts = {}
coh_counts(H_Coh_container, H_Doc_container, H_Coh_narrative_counts, H_Coh_no_narrative_counts, H_Coh_argument_counts, H_Coh_no_argument_counts)

### Grover coherence counts
G_Coh_no_narrative_counts = {}
G_Coh_narrative_counts = {}
G_Coh_no_argument_counts = {}
G_Coh_argument_counts = {}
coh_counts(G_Coh_container, G_Doc_container, G_Coh_narrative_counts, G_Coh_no_narrative_counts, G_Coh_argument_counts, G_Coh_no_argument_counts)


### Clean up and rename coherence relations separated by human origin, narrativity, and argumentation
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

################################################################################

### Average lengths of coherence relations, divided by narrativity & arguments

### ADAM's Thematic Unity Hypothesis: documents marked as containing a `narrative`
# should have more longer-range coherence relations, suggesting they are talking
# about a central topic

### function that gets lengths of relations for narrative and non-narrative docs
def get_coh_lengths(mode, Coh_container, Doc_container, no_lengths, lengths):
    for annotator in Coh_container:
        for number in Coh_container[annotator]:
            for relation in Coh_container[annotator][number]:
                try:
                    tmp = [int(i) for i in relation[0:4]]
                    min = np.min(tmp)
                    max = np.max(tmp)
                    doc_level_index = 3 if mode=="narrative" else 12 if mode=="argument" else 999
                    if Doc_container[annotator][number][doc_level_index] != '0':
                        if relation[4] in lengths:
                            lengths[relation[4]].append(max-min)
                        else:
                            diff = max-min
                            narrative_lengths[relation[4]] = [diff]
                    else:
                        if relation[4] in no_lengths:
                            no_lengths[relation[4]].append(max-min)
                        else:
                            diff = max-min
                            no_lengths[relation[4]] = [diff]
                except: # because sometimes there's ? instead of a segment number
                    pass


### Get coherence relation lengths for human docs, narrative
H_narrative_lengths = {}
H_no_narrative_lengths = {}
get_coh_lengths("narrative", H_Coh_container, H_Doc_container, H_no_narrative_lengths, H_narrative_lengths)

### get coherence relation lengths for grover docs, narrative
G_narrative_lengths = {}
G_no_narrative_lengths = {}
get_coh_lengths("narrative", G_Coh_container, G_Doc_container, G_no_narrative_lengths, G_narrative_lengths)

### get coherence relation lengths for human docs, argumentation
H_argument_lengths = {}
H_no_argument_lengths = {}
get_coh_lengths("argument", H_Coh_container, H_Doc_container, H_no_argument_lengths, H_argument_lengths)

### get coherence relation lengths for grover docs, argumentation
G_argument_lengths = {}
G_no_argument_lengths = {}
get_coh_lengths("argument", G_Coh_container, G_Doc_container, G_no_argument_lengths, G_argument_lengths)


### function that cleans up coherence relation lengths 
def clean_up_coh_lengths(old_dict, clean_dict):
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

### define clean dictionaries for narratives
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
G_Coh_narrative_lengths_clean = copy.deepcopy(H_Coh_narrative_lengths_clean)
G_Coh_no_narrative_lengths_clean = copy.deepcopy(H_Coh_narrative_lengths_clean)


### clean dictionaries for arguments 
H_Coh_argument_lengths_clean = copy.deepcopy(H_Coh_narrative_lengths_clean)
H_Coh_no_argument_lengths_clean = copy.deepcopy(H_Coh_narrative_lengths_clean)
G_Coh_argument_lengths_clean = copy.deepcopy(H_Coh_narrative_lengths_clean)
G_Coh_no_argument_lengths_clean = copy.deepcopy(H_Coh_narrative_lengths_clean)


### clean up coh_lengths for narratives
clean_up_coh_lengths(G_narrative_lengths,G_Coh_narrative_lengths_clean)
clean_up_coh_lengths(G_no_narrative_lengths,G_Coh_no_narrative_lengths_clean)
clean_up_coh_lengths(H_narrative_lengths,H_Coh_narrative_lengths_clean)
clean_up_coh_lengths(H_no_narrative_lengths,H_Coh_no_narrative_lengths_clean)


### clean up coh_lengths for arguments
clean_up_coh_lengths(H_argument_lengths,H_Coh_argument_lengths_clean)
clean_up_coh_lengths(H_no_argument_lengths,H_Coh_no_argument_lengths_clean)
clean_up_coh_lengths(G_argument_lengths,G_Coh_argument_lengths_clean)
clean_up_coh_lengths(G_no_argument_lengths,G_Coh_no_argument_lengths_clean)


### helper that calculates average coherence relation length, given dict of coh lengths
def avg_coh_length(avg_lengths, lengths_clean):
    for relation in lengths_clean:
        avg_lengths[relation] = np.mean(lengths_clean[relation])


### Calculate average coherence relation lengths - narrativity, human and grover
H_Coh_narrative_avg_lengths = {}
avg_coh_length(H_Coh_narrative_avg_lengths, H_Coh_narrative_lengths_clean)

H_Coh_no_narrative_avg_lengths = {}
avg_coh_length(H_Coh_no_narrative_avg_lengths, H_Coh_no_narrative_lengths_clean)

G_Coh_narrative_avg_lengths = {}
avg_coh_length(G_Coh_narrative_avg_lengths, G_Coh_narrative_lengths_clean)

G_Coh_no_narrative_avg_lengths = {}
avg_coh_length(G_Coh_no_narrative_avg_lengths, G_Coh_no_narrative_lengths_clean)


### Calculate average coherence relation lengths - argumentation, human and grover
H_Coh_argument_avg_lengths = {}
avg_coh_length(H_Coh_argument_avg_lengths, H_Coh_argument_lengths_clean)

H_Coh_no_argument_avg_lengths = {}
avg_coh_length(H_Coh_no_argument_avg_lengths, H_Coh_no_argument_lengths_clean)

G_Coh_argument_avg_lengths = {}
avg_coh_length(G_Coh_argument_avg_lengths, G_Coh_argument_lengths_clean)

G_Coh_no_argument_avg_lengths = {}
avg_coh_length(G_Coh_no_argument_avg_lengths, G_Coh_no_argument_lengths_clean)


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

### function that looks at whether the last statement has a generic main referent, and whether reports have more coerced states
def last_generic_coerced_report(SE_container, Doc_container, narr_total_counter, last_gen_counter, rep_coerced_counter):
    for annotator in SE_container:
        for number in SE_container[annotator]:
            if Doc_container[annotator][number][3] != "0" and Doc_container[annotator][number][12] != "0":
                narr_total_counter['narrative & argument'] += 1
                if "GENERIC" in SE_container[annotator][number][len(SE_container[annotator][number])-1] or "GNERIC" in SE_container[annotator][number][len(SE_container[annotator][number])-1]:
                    last_gen_counter['narrative & argument'] += 1

                for SE in SE_container[annotator][number]:
                    if "COERCED" in SE:
                        rep_coerced_counter['narrative & argument'] += 1

            elif Doc_container[annotator][number][3] != "0" and Doc_container[annotator][number][12] == "0":
                narr_total_counter['narrative'] += 1
                if "GENERIC" in SE_container[annotator][number][len(SE_container[annotator][number])-1] or "GNERIC" in SE_container[annotator][number][len(SE_container[annotator][number])-1]:
                    last_gen_counter['narrative'] += 1

                for SE in SE_container[annotator][number]:
                    if "COERCED" in SE:
                        rep_coerced_counter['narrative'] += 1

            elif Doc_container[annotator][number][3] == "0" and Doc_container[annotator][number][12] != "0":
                narr_total_counter['argument'] += 1
                if "GENERIC" in SE_container[annotator][number][len(SE_container[annotator][number])-1] or "GNERIC" in SE_container[annotator][number][len(SE_container[annotator][number])-1]:
                    last_gen_counter['argument'] += 1

                for SE in SE_container[annotator][number]:
                    if "COERCED" in SE:
                        rep_coerced_counter['argument'] += 1

            elif Doc_container[annotator][number][3] == "0" and Doc_container[annotator][number][12] == "0":
                narr_total_counter['no narrative or argument'] += 1
                if "GENERIC" in SE_container[annotator][number][len(SE_container[annotator][number])-1] or "GNERIC" in SE_container[annotator][number][len(SE_container[annotator][number])-1]:
                    last_gen_counter['no narrative or argument'] += 1

                for SE in SE_container[annotator][number]:
                    if "COERCED" in SE:
                        rep_coerced_counter['no narrative or argument'] += 1

    for key in last_gen_counter:
        last_gen_counter[key] = last_gen_counter[key] / narr_total_counter[key]

### Last generic and coerced report hypotheses - Human
H_last_gen_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}
H_narr_total_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}
H_rep_coerced_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}
last_generic_coerced_report(H_SE_container, H_Doc_container, H_narr_total_counter, H_last_gen_counter, H_rep_coerced_counter)

### Last generic and coerced report hypotheses - Grover
G_last_gen_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}
G_narr_total_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}
G_rep_coerced_counter = {'narrative':0, 'argument': 0, 'narrative & argument': 0, 'no narrative or argument':0}
last_generic_coerced_report(G_SE_container, G_Doc_container, G_narr_total_counter, G_last_gen_counter, G_rep_coerced_counter)


print("***")
print("Human")
print("***")
print("Total count")
print(H_narr_total_counter)
print("Last generic")
print(H_last_gen_counter)
print("Coerced report")
print(H_rep_coerced_counter)

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

plt.bar(G_incoherent_rels.keys(), G_incoherent_rels.values(), color='g')
plt.xticks(list(G_incoherent_rels.keys()),list(G_incoherent_rels.keys()), rotation='vertical',fontsize=14)
plt.axis(ymax=50)
plt.xlabel("Incoherent Coherence Relation", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.title('All docs - Grover', fontsize=20)
plt.tight_layout()
plt.show()

G_incoherent_rels.pop('Repetition')
G_incoherent_rels.pop('Degenerate')
for key in G_incoherent_rels:
    G_incoherent_rels[key] = G_incoherent_rels[key] / G_coherent_rels[key]

### Proportion plot of incoherent relations - Grover (excluding rep, deg)

plt.bar(G_incoherent_rels.keys(), G_incoherent_rels.values(), color='g')
plt.xticks(list(G_incoherent_rels.keys()),list(G_incoherent_rels.keys()), rotation='vertical',fontsize=14)
plt.axis(ymax=0.7)
plt.xlabel("Incoherent Coherence Relation", fontsize=18)
plt.ylabel("Proportion of relevant relation in texts", fontsize=18)
plt.title('All docs - Grover', fontsize=20)
plt.tight_layout()
plt.show()
