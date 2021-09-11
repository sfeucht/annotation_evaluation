# produces correlations for situation entities vs narrative presence (line 102), SE vs argument presence (192)
# SE vs argument quality measures (282), and SE vs narrative quality measures (375)

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

SE_types = ['BOUNDED EVENT (SPECIFIC)', 'BOUNDED EVENT (GENERIC)', 'UNBOUNDED EVENT (SPECIFIC)',
            'BASIC STATE', 'COERCED STATE (SPECIFIC)', 'COERCED STATE (GENERIC)', 'PERFECT COERCED STATE (SPECIFIC)',
            'PERFECT COERCED STATE (GENERIC)', 'GENERIC SENTENCE (HABITUAL)', 'GENERIC SENTENCE (STATIC)',
            'GENERIC SENTENCE (DYNAMIC)', 'GENERALIZING SENTENCE (DYNAMIC)',
            'QUESTION', 'OTHER']

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
H_SE_container = annotator_tag(H_SE_container)
G_SE_container = annotator_tag(G_SE_container)
D_SE_container = annotator_tag(D_SE_container)

########################################################
### SE FREQUENCY'S CORRELATION WITH PRESENCE OF NARRATIVE -> instructions for running function are below
# p_of_nar_correlation_calculator (starting line 178)

def p_of_narrative_SE_counts(Doc_container, SE_container, SE_type): # produces a dictionary with counts of a specific SE type
    # from each document, called by p_of_nar_correlation_calculator

    quality_dict = {}

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            for SE_id in SE_container[annotator].keys():

                SE_list = SE_container[annotator][doc_id]

                if doc_id == SE_id: # to match IDs across dicts

                    for SE in SE_list:
                        if(SE == SE_type):
                            if doc_id not in quality_dict: # if doc_id not in quality_dict, creates new entry for it
                                quality_dict[doc_id] = {}
                            if(SE not in quality_dict[doc_id]): # adds counter for each SE type for each doc_id
                                quality_dict[doc_id][SE] = 0
                            quality_dict[doc_id][SE] += 1
                    if (SE_type not in SE_list): # so SE types that don't appear in doc aren't left out of the dict
                        if doc_id not in quality_dict:
                            quality_dict[doc_id] = {}
                        if(SE not in quality_dict[doc_id]):
                            quality_dict[doc_id][SE_type] = 0
    return quality_dict

def presence_of_narrative(Doc_container, index): # produces a dictionary of narrative presence for each document (with 0 for
    # no narrative and 1 for some narrative), called by p_of_nar_correlation_calculator

    rating_dict = {}

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            ratings = Doc_container[annotator][doc_id]

            if (ratings[index] == 'NA') or (ratings[index] == '0'): # adds 0 to dict if there's no narrative
                if(doc_id not in rating_dict):
                    rating_dict[doc_id] = {}
                    rating_dict[doc_id] = 0
            if (ratings[index] != 'NA') and (ratings[index] != '0'): # adds 1 to dict if there's some narrative
                if(doc_id not in rating_dict):
                    rating_dict[doc_id] = {}
                    rating_dict[doc_id] = 1

    return rating_dict

def p_of_nar_correlation_calculator(Doc_container, SE_container): # uses both of the previous functions to find
    # correlations between each SE type and presence of narrative

    for SE in SE_types:
        quality_dict = p_of_narrative_SE_counts(Doc_container, SE_container, SE)
        rating_dict = presence_of_narrative(Doc_container, 3)
        quality_list = []
        rating_list = []

        for k,v in rating_dict.items(): # creates list of 0s and 1s to be fed into pointbiserialr
            rating_list.append(int(v))

        for k1,v1 in quality_dict.items(): # creates list of SE counts to be fed into pointbiserialr
            entry = v1

            for k2,v2 in entry.items():
                quality_list.append(v2)

        r,p = scipy.stats.pointbiserialr(rating_list, quality_list)

        if abs(r) > 0.2:
            print(SE, 'FREQUENCY CORRELATED WITH NARRATIVE PRESENCE:', 'r =', round(r, 2), ', p =', round(p, 2))

    return

#Uncomment for grover results
print(p_of_nar_correlation_calculator(G_Doc_container, G_SE_container))
#based on 180 grover documents

#Uncomment for GPT-3 results
#print(p_of_nar_correlation_calculator(D_Doc_container, D_SE_container))
#based on 72 davinci documents

#Uncomment for human results
#print(p_of_nar_correlation_calculator(H_Doc_container, H_SE_container))
#based on 192 human documents

########################################################

########################################################
### SE FREQUENCY'S CORRELATION WITH PRESENCE OF ARGUMENT -> instructions for running function are below
# p_of_arg_correlation_calculator (starting at line 268)

def p_of_argument_SE_counts(Doc_container, SE_container, SE_type): # produces a dictionary with counts of a specific SE type
    # from each document, called by p_of_arg_correlation_calculator

    quality_dict = {}

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            for SE_id in SE_container[annotator].keys():

                SE_list = SE_container[annotator][doc_id]

                if doc_id == SE_id: # to match IDs across dicts

                    for SE in SE_list:
                        if(SE == SE_type):
                            if doc_id not in quality_dict: # if doc_id not in quality_dict, creates new entry for it
                                quality_dict[doc_id] = {}
                            if(SE not in quality_dict[doc_id]): # adds counter for each SE type for each doc_id
                                quality_dict[doc_id][SE] = 0
                            quality_dict[doc_id][SE] += 1
                    if (SE_type not in SE_list): # so SE types that don't appear in doc aren't left out of the dict
                        if doc_id not in quality_dict:
                            quality_dict[doc_id] = {}
                        if(SE not in quality_dict[doc_id]):
                            quality_dict[doc_id][SE_type] = 0
    return quality_dict

def presence_of_argument(Doc_container, index): # produces a dictionary of argument presence for each document (with 0 for
    # no argument and 1 for some argument), called by p_of_arg_correlation_calculator

    rating_dict = {}

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            ratings = Doc_container[annotator][doc_id]

            if (ratings[index] == 'NA') or (ratings[index] == '0'): # adds 0 to dict if there's no narrative
                if(doc_id not in rating_dict):
                    rating_dict[doc_id] = {}
                    rating_dict[doc_id] = 0
            if (ratings[index] != 'NA') and (ratings[index] != '0'): # adds 1 to dict if there's some narrative
                if(doc_id not in rating_dict):
                    rating_dict[doc_id] = {}
                    rating_dict[doc_id] = 1

    return rating_dict

def p_of_arg_correlation_calculator(Doc_container, SE_container): # uses both of the previous functions to find
    # correlations between each SE type and presence of argument

    for SE in SE_types:
        quality_dict = p_of_argument_SE_counts(Doc_container, SE_container, SE)
        rating_dict = presence_of_argument(Doc_container, 12)
        quality_list = []
        rating_list = []

        for k,v in rating_dict.items(): # creates list of 0s and 1s to be fed into pointbiserialr
            rating_list.append(int(v))

        for k1,v1 in quality_dict.items(): # creates list of SE counts to be fed into pointbiserialr
            entry = v1

            for k2,v2 in entry.items():
                quality_list.append(v2)

        r,p = scipy.stats.pointbiserialr(rating_list, quality_list)

        if abs(r) > 0.2:
            print(SE, 'FREQUENCY CORRELATED WITH ARGUMENT PRESENCE:', 'r =', round(r, 2), ', p =', round(p, 2))

    return

#Uncomment for grover results
#print(p_of_arg_correlation_calculator(G_Doc_container, G_SE_container))
#based on 180 grover documents

#Uncomment for GPT-3 results
#print(p_of_nar_correlation_calculator(D_Doc_container, D_SE_container))
#based on 72 davinci documents

#Uncomment for grover results
#print(p_of_arg_correlation_calculator(H_Doc_container, H_SE_container))
#based on 192 human documents

########################################################

########################################################
### SE FREQUENCY'S CORRELATION WITH ARGUMENT QUALITY MEASURES -> instructions for running function are below
# arg_correlation_calculator (starting line 361)

def argument_SE_counts(Doc_container, SE_container, SE_type, index): # produces a dictionary with counts of a specific SE type
# from each document, called by arg_correlation_calculator

    quality_dict = {}

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            for SE_id in SE_container[annotator].keys():
                ratings = Doc_container[annotator][doc_id]
                SE_list = SE_container[annotator][doc_id]

                if doc_id == SE_id: # to match IDs across dicts

                    if (ratings[index] != 'NA') and (ratings[index] != '0'):
                        for SE in SE_list:
                            if(SE == SE_type):
                                if doc_id not in quality_dict: # if doc_id not in quality_dict, creates new entry for it
                                    quality_dict[doc_id] = {}
                                if(SE not in quality_dict[doc_id]): # adds counter for each SE type for each doc_id
                                    quality_dict[doc_id][SE] = 0
                                quality_dict[doc_id][SE] += 1
                        if (SE_type not in SE_list): # so SE types that don't appear in doc aren't left out of the dict
                            if doc_id not in quality_dict:
                                quality_dict[doc_id] = {}
                            if(SE not in quality_dict[doc_id]):
                                quality_dict[doc_id][SE_type] = 0
    return quality_dict

def argument_ratings(Doc_container, index): # produces a list with the rating number of a given quality measure from
    # each from each document, called by arg_correlation_calculator

    rating_dict = {}

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            ratings = Doc_container[annotator][doc_id]

            if (ratings[index] != 'NA') and (ratings[index] != '0'): # if doc_id not in rating_dict, creates new entry for it
                if(doc_id not in rating_dict):
                    rating_dict[doc_id] = {}
                if(ratings[index] not in rating_dict[doc_id]): # adds rating to entry
                    rating_dict[doc_id] = ratings[index]

    return rating_dict

def arg_correlation_calculator(Doc_container, SE_container): # uses both of the previous functions to find correlations
    # between each SE type and each quality measure in all of the documents with argument
    indices = [13,14]

    for index in indices:
        for SE in SE_types:
            quality_dict = argument_SE_counts(Doc_container, SE_container, SE, index)
            rating_dict = argument_ratings(Doc_container, index)
            quality_list = []
            rating_list = []

            for k,v in rating_dict.items(): # creates list of quality ratings to be fed into pearsonr
                rating_list.append(int(v))

            for k1,v1 in quality_dict.items(): # creates list of SE counts to be fed into pearsonr
                entry = v1

                for k2,v2 in entry.items():
                    quality_list.append(v2)

            r,p = pearsonr(quality_list, rating_list)

            if index == 13 and abs(r) > 0.2:
                print(SE, 'FREQUENCY CORRELATED WITH COGENCY:', 'r = ', round(r, 2), 'p = ', round(p, 2))

            if index == 14 and abs(r) > 0.2:
                print(SE, 'FREQUENCY CORRELATED WITH EFFECTIVENESS:', round(r, 2), 'p = ', round(p, 2))

    return

#Uncomment for grover results
#print(arg_correlation_calculator(G_Doc_container, G_SE_container))
#based on 71 grover documents

#Uncomment for GPT-3 results
#print(arg_correlation_calculator(D_Doc_container, D_SE_container))
#based on 52 davinci documents

#Uncomment for human results
#print(arg_correlation_calculator(H_Doc_container, H_SE_container))
#based on 115 human documents

########################################################

########################################################
### SE FREQUENCY'S CORRELATION WITH NARRATIVE QUALITY MEASURES -> instructions for running function are below
# nar_correlation_calculator (starting line 459)

def narrative_SE_counts(Doc_container, SE_container, SE_type, index): # produces a dictionary with counts of a specific SE type
    # from each document, called by nar_correlation_calculator

    quality_dict = {}

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            for SE_id in SE_container[annotator].keys():
                ratings = Doc_container[annotator][doc_id]
                SE_list = SE_container[annotator][doc_id]

                if doc_id == SE_id: # to match IDs across dicts

                    if (ratings[index] != 'NA') and (ratings[index] != '0'):

                        for SE in SE_list:
                            if(SE == SE_type):
                                if doc_id not in quality_dict: # if doc_id not in quality_dict, creates new entry for it
                                    quality_dict[doc_id] = {}
                                if(SE not in quality_dict[doc_id]): # adds counter for each SE type for each doc_id
                                    quality_dict[doc_id][SE] = 0
                                quality_dict[doc_id][SE] += 1
                        if (SE_type not in SE_list): # so SE types that don't appear in doc aren't left out of the dict
                            if doc_id not in quality_dict:
                                quality_dict[doc_id] = {}
                            if(SE not in quality_dict[doc_id]):
                                quality_dict[doc_id][SE_type] = 0
    return quality_dict

def narrative_ratings(Doc_container, index): # produces a dictionary with the rating number of a given quality measure from
    # each from each document, called by nar_correlation_calculator

    rating_dict = {}

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            ratings = Doc_container[annotator][doc_id]

            if (ratings[index] != 'NA') and (ratings[index] != '0'): # if doc_id not in rating_dict, creates new entry for it
                if(doc_id not in rating_dict):
                    rating_dict[doc_id] = {}
                if(ratings[index] not in rating_dict[doc_id]): # adds rating to entry
                    rating_dict[doc_id] = ratings[index]
    return rating_dict

def nar_correlation_calculator(Doc_container, SE_container): # uses both of the previous functions to find correlations
    # between each SE type and each quality measure in all of the documents with narrative
    indices = [4,5,6,7]

    for index in indices:
        for SE in SE_types:
            quality_dict = narrative_SE_counts(Doc_container, SE_container, SE, index)
            rating_dict = narrative_ratings(Doc_container, index)
            quality_list = []
            rating_list = []

            for k,v in rating_dict.items(): # creates list of quality ratings to be fed into pearsonr
                rating_list.append(int(v))

            for k1,v1 in quality_dict.items(): # creates list of SE counts to be fed into pearsonr
                entry = v1

                for k2,v2 in entry.items():
                    quality_list.append(v2)

            r,p = scipy.stats.pearsonr(quality_list, rating_list)

            if index == 4 and abs(r) > 0.2:
                print(SE, 'FREQUENCY CORRELATED WITH PLAUSIBILITY:', 'r = ', round(r, 2), 'p = ', round(p, 2))

            if index == 5 and abs(r) > 0.2:
                print(SE, 'FREQUENCY CORRELATED WITH COMPLETENESS:', 'r = ', round(r, 2), 'p = ', round(p, 2))

            if index == 6 and abs(r) > 0.2:
                print(SE, 'FREQUENCY CORRELATED WITH CONSISTENCY:', 'r = ', round(r, 2), 'p = ', round(p, 2))

            if index == 7 and abs(r) > 0.2:
                print(SE, 'FREQUENCY CORRELATED WITH COVERAGE:', 'r = ', round(r, 2), 'p = ', round(p, 2))
    return

#Uncomment for grover results
#print(nar_correlation_calculator(G_Doc_container, G_SE_container))
#based on 78 grover documents

#Uncomment for GPT-3 results
#print(nar_correlation_calculator(D_Doc_container, D_SE_container))
#based on 24 davinci documents

#Uncomment for grover results
#print(nar_correlation_calculator(H_Doc_container, H_SE_container))
#based on 78 human documents

########################################################
