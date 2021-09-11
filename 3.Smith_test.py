# -*- coding: utf-8 -*-

# TODO: adjudicate between annotators. Currently we're taking only one for each
# shared document

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
from clean_up_SE_coh import simplify_all_SE_types
from clean_up_SE_coh import clean_up_coh_rels
from extract_annotations import fill_in_human_grover, fill_in_containers

### determine path to the annotations
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"

### Extract and print out document-annotator assignments
regex = re.compile('[^a-zA-Z]')
annotators = {"Sheridan":[],"Muskaan":[],"Kate":[]}
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

### Extract the human or Grover source of each document

# Create lists for keeping track of human and Grover generations
h_docs = []
g_docs = []
fill_in_human_grover(h_docs, g_docs)

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

doc_counter = fill_in_containers(h_docs, g_docs, G_SE_container, G_Coh_container, 
G_Doc_container, H_SE_container, H_Coh_container, H_Doc_container, SE_accounted_for,
Coh_accounted_for, doc_counter)

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

plt.bar(H_narrativity_counts.keys(), H_narrativity_counts.values(), color='b')
plt.xticks(list(H_narrativity_counts.keys()),list(H_narrativity_counts.keys()), rotation='vertical',fontsize=14)
plt.axis(ymax=35)
plt.xlabel("Proportion narrative in a document \n(0: none; 5: all)",fontsize=18)
plt.ylabel("Frequency",fontsize=18)
plt.title('Human-generated documents',fontsize=20)
plt.tight_layout()
plt.show()

plt.bar(G_narrativity_counts.keys(), G_narrativity_counts.values(), color='g')
plt.xticks(list(G_narrativity_counts.keys()),list(G_narrativity_counts.keys()), rotation='vertical',fontsize=14)
plt.axis(ymax=35)
plt.xlabel("Proportion narrative in a document \n(0: none; 5: all)",fontsize=18)
plt.ylabel("Frequency",fontsize=18)
plt.title('Grover-generated documents',fontsize=20)
plt.tight_layout()
plt.show()


################################################################################

### Clean up and rename Situation Entity annotations - Human and Grover

# HYPOTHESIS: narrative documents should have higher avg bounded events and states
# average up for bounded events and states for both dictionaries

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
simplify_all_SE_types(H_no_narrative_counts, H_no_narrative_counts_clean)
simplify_all_SE_types(H_narrative_counts, H_narrative_counts_clean)

### clean up grover no narrative and narrative SE counts
simplify_all_SE_types(G_no_narrative_counts, G_no_narrative_counts_clean)
simplify_all_SE_types(G_narrative_counts, G_narrative_counts_clean)


### function that takes two dicts with identical keys and plots them with given colors
def two_bar_dict_plot(dict1, dict2, label1, label2, color1, color2, title, ylabel):
    x = np.arange(len(dict1.keys()))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - width/2, dict1.values(), width, label=label1, color=color1)
    ax.bar(x + width/2, dict2.values(), width, label=label2, color=color2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(dict1.keys(), rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.show()


### Plot Situation Entity distribution divided by narrativity - Human
two_bar_dict_plot(H_narrative_counts_clean, H_no_narrative_counts_clean, 
"Narrative", "No Narrative", "m", "gray", "SE Types for Human Documents", "Proportion of SE types")

### Plot Situation Entity distribution divided by narrativity - Grover
two_bar_dict_plot(G_narrative_counts_clean, G_no_narrative_counts_clean, 
"Narrative", "No Narrative", "m", "gray", "SE Types for Grover Documents", "Proportion of SE types")


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
two_bar_dict_plot(H_argument_counts_clean, H_no_argument_counts_clean, 
"Argument", "No Argument", "r", "gray", "SE Types for Human Documents", "Proportion of SE types")

### Plot Situation Entity distribution divided by argumentation - Grover
two_bar_dict_plot(G_argument_counts_clean, G_no_argument_counts_clean, 
"Argument", "No Argument", "r", "gray", "SE Types for Grover Documents", "Proportion of SE types")


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
two_bar_dict_plot(H_Coh_narrative_counts_clean, H_Coh_no_narrative_counts_clean, 
"Narrative", "No Narrative", "m", "gray", "Coherence Relation Counts for Human Documents", "Frequency of Relations")

### Plot the distribution of coherence relations divided by narrativity - Grover
two_bar_dict_plot(G_Coh_narrative_counts_clean, G_Coh_no_narrative_counts_clean, 
"Narrative", "No Narrative", "m", "gray", "Coherence Relation Counts for Grover Documents", "Frequency of Relations")

### Plot the distribution of coherence relations divided by argumentation - Human
two_bar_dict_plot(H_Coh_argument_counts_clean, H_Coh_no_argument_counts_clean, 
"Argument", "No Argument", "r", "gray", "Coherence Relation Counts for Human Documents", "Frequency of Relations")

### Plot the distribution of coherence relations divided by argumentation - Grover
two_bar_dict_plot(G_Coh_argument_counts_clean, G_Coh_no_argument_counts_clean, 
"Argument", "No Argument", "r", "gray", "Coherence Relation Counts for Grover Documents", "Frequency of Relations")


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
two_bar_dict_plot(H_Coh_narrative_counts_clean, H_Coh_no_narrative_counts_clean, 
"Narrative", "No Narrative", "m", "gray", "Coherence Relation Counts for Human Documents", "Proportion of Relations")

### Plot the proportion of coherence relations divided by narrativity - Grover
two_bar_dict_plot(G_Coh_narrative_counts_clean, G_Coh_no_narrative_counts_clean, 
"Narrative", "No Narrative", "m", "gray", "Coherence Relation Counts for Grover Documents", "Proportion of Relations")

### Plot the proportion of coherence relations divided by argumentation - Human
two_bar_dict_plot(H_Coh_argument_counts_clean, H_Coh_no_argument_counts_clean, 
"Argument", "No Argument", "r", "gray", "Coherence Relation Counts for Human Documents", "Proportion of Relations")

### Plot the proportion of coherence relations divided by argumentation - Grover
two_bar_dict_plot(G_Coh_argument_counts_clean, G_Coh_no_argument_counts_clean, 
"Argument", "No Argument", "r", "gray", "Coherence Relation Counts for Grover Documents", "Proportion of Relations")


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
                    doc_level_index = 3 if mode=="narrative" else 12 if mode=="argument" else Exception("mode must be 'narrative' or 'argument'")
                    if Doc_container[annotator][number][doc_level_index] != '0':
                        if relation[4] in lengths:
                            lengths[relation[4]].append(max-min)
                        else:
                            diff = max-min
                            lengths[relation[4]] = [diff]
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


### Plot average coherence relation lengths for human docs - narrative vs non-narrative
two_bar_dict_plot(H_Coh_narrative_avg_lengths, H_Coh_no_narrative_avg_lengths, 
"Narrative", "No Narrative", "m", "gray",'Coherence Relation Lengths for Human Documents','Average length (segments)')

### Plot average coherence relation lengths for Grover docs - narrative vs non-narrative
two_bar_dict_plot(G_Coh_narrative_avg_lengths, G_Coh_no_narrative_avg_lengths, 
"Narrative", "No Narrative", "m", "gray",'Coherence Relation Lengths for Grover Documents','Average length (segments)')

### Plot average coherence relation lengths divided by argumentation - Human
two_bar_dict_plot(H_Coh_argument_avg_lengths, H_Coh_no_argument_avg_lengths, 
"Argument", "No Argument", "r", "gray", "Coherence Relation Lengths for Human Documents", "Average length (segments)")

### Plot average coherence relation lengths divided by argumentation - Grover
two_bar_dict_plot(G_Coh_argument_avg_lengths, G_Coh_no_argument_avg_lengths, 
"Argument", "No Argument", "r", "gray", "Coherence Relation Lengths for Grover Documents", "Average length (segments)")

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

### Distribution of incoherent and coherent relations - Human and Grover

### function that counts up incoherent and coherent relations
### returns 
def count_up_incoherence(Coh_container, incoherent_rels, coherent_rels, incoherent_counter):
    doc_counter = 0
    for annotator in Coh_container:
        for number in Coh_container[annotator]:
            doc_counter += 1
            indicator = 0
            for relation in Coh_container[annotator][number]:

                if relation[4] == 'cex':
                    incoherent_rels['Cause/effect'] += 1
                    indicator = 1
                elif relation[4] in ['ce', 'cew', 'cer']:
                    coherent_rels['Cause/effect'] += 1

                elif relation[4] == 'elabx':
                    incoherent_rels['Elaboration'] += 1
                    indicator = 1
                elif relation[4] in ['elab', 'ealb', 'elav', 'elabl']:
                    coherent_rels['Elaboration'] += 1

                elif relation[4] == 'samex':
                    incoherent_rels['Same'] += 1
                    indicator = 1
                elif relation[4] == 'same':
                    coherent_rels['Same'] += 1

                elif relation[4] == 'attrx':
                    incoherent_rels['Attribution'] += 1
                    indicator = 1
                elif relation[4] in ['attrm','attr']:
                    coherent_rels['Attribution'] += 1

                elif relation[4] in ['deg','degenerate','mal']:
                    incoherent_rels['Degenerate'] += 1
                    indicator = 1

                elif relation[4] == 'simx':
                    incoherent_rels['Similarity'] += 1
                    indicator = 1
                elif relation[4] == 'sim':
                    coherent_rels['Similarity'] += 1

                elif relation[4] == 'contrx':
                    incoherent_rels['Contrast'] += 1
                    indicator = 1
                elif relation[4] == 'contr':
                    coherent_rels['Contrast'] += 1

                elif relation[4] == 'tempx':
                    incoherent_rels['Temporal sequence'] += 1
                    indicator = 1
                elif relation[4] == 'temp':
                    coherent_rels['Temporal sequence'] += 1

                elif relation[4] == 'vex':
                    incoherent_rels['Violated expectation'] += 1
                    indicator = 1
                elif relation[4] == 've':
                    coherent_rels['Violated expectation'] += 1

                elif relation[4] == 'exampx':
                    incoherent_rels['Example'] += 1
                    indicator = 1
                elif relation[4] == 'examp':
                    coherent_rels['Example'] += 1

                elif relation[4] == 'condx':
                    incoherent_rels['Condition'] += 1
                    indicator = 1
                elif relation[4] == 'cond':
                    coherent_rels['Condition'] += 1

                elif relation[4] == 'genx':
                    incoherent_rels['Generalization'] += 1
                    indicator = 1
                elif relation[4] == 'gen':
                    coherent_rels['Generalization'] += 1

                elif relation[4] == 'rep':
                    incoherent_rels['Repetition'] += 1
                    indicator = 1
            if indicator == 1:
                incoherent_counter += 1
            
    return doc_counter

### apply function to human-generated docs
H_incoherent_counter = 0
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
H_doc_counter = count_up_incoherence(H_Coh_container, H_incoherent_rels, H_coherent_rels, H_incoherent_counter)


### apply function to grover docs
G_incoherent_counter = 0
G_incoherent_rels = copy.deepcopy(H_incoherent_rels)
G_coherent_rels = copy.deepcopy(H_incoherent_rels)
G_doc_counter = count_up_incoherence(G_Coh_container, G_incoherent_rels, G_coherent_rels, G_incoherent_counter)


### print human incoherence results
print("***")
print("Distribution of Incoherent Relations - Human")
print(H_incoherent_rels)
print(H_incoherent_counter / H_doc_counter)

### print grover incoherence results 
print("***")
print("Distribution of Incoherent Relations - Grover")
print(G_incoherent_rels)
print(G_incoherent_counter / G_doc_counter)

'''
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
'''
