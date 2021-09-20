# run to create a graph comparing the proportion of coherence relations that are incoherent/ nonsensical across grover
# and davinci documents

import csv
import re
import os
import sys

import scipy
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
annotators = {"0":[],"3":[],"1":[]}
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

### Extract the human or Grover source of each document

# Create lists for keeping track of human and Grover generations
h_docs = []
g_docs = []
d_docs = []
fill_in_human_grover(h_docs, g_docs, d_docs)

### EXTRACT THE HUMAN, GROVER, OR DAVINCI SOURCE OF EACH DOCUMENT

# Create lists for keeping track of human and AI generations
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
D_Doc_container, SE_accounted_for,
Coh_accounted_for, doc_counter)

Coh_types = ['elab', 'temp', 've', 'ce', 'same', 'contr', 'sim', 'attr', 'examp', 'cond', 'gen']
nons_Coh_types = ['elabx', 'tempx', 'vex', 'cex', 'samex', 'contrx', 'simx', 'attrx', 'exampx', 'condx', 'genx']

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

########################################################
### NONSENSICAL COHERENCE RELATIONS PROPORTION GENERATOR

def Coh_type_counter(Doc_container, Coh_container, type_list): # produces a dictionary with the number of each COH type
# given a list of COH types, called by proportion_generator

    count_dict = {}

    for type in type_list:

        Coh_count = 0

        for annotator in Doc_container.keys():
            for doc_id in Doc_container[annotator].keys():
                for Coh_id in Coh_container[annotator].keys():

                    Coh_lists = Coh_container[annotator][doc_id]

                    if doc_id == Coh_id: # to match IDs across dicts

                        for list in Coh_lists:

                            if(list[4] == type): # count the number of times a given COH appears in the Coh_container
                                Coh_count += 1

        count_dict[type] = Coh_count # create dictionary (key = COH type, value = count)

    return count_dict

def proportion_generator(Doc_container, Coh_container): # produces a dictionary with the proportion of nonsensical
# relations for each Coh type

    sens_dict = Coh_type_counter(Doc_container, Coh_container, Coh_types) # one dict for coherent COHs
    nons_dict = Coh_type_counter(Doc_container, Coh_container, nons_Coh_types) # one dict for nonsensical relations

    prop_dict = {}
    prop_list = []

    for k,v in nons_dict.items(): # calculate the proportion of Coh type that are incoherent/ nonsensical, add to list
        for k1,v1 in sens_dict.items():
            if k1 in k:

                prop_list.append(round((v / (v + v1)), 2))

    prop_dict['Proportions'] = prop_list # list of proportions from prop_dict will be used in creating graph below
    prop_dict['Coherence Relations'] = Coh_types

    return prop_dict

# setup for graph function

grover_dict = proportion_generator(G_Doc_container, G_Coh_container)
davinci_dict = proportion_generator(D_Doc_container, D_Coh_container)

grover_props = grover_dict['Proportions']
davinci_props = davinci_dict['Proportions']

ind = np.arange(len(grover_props))  # the x locations for the groups
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, grover_props, width, #add rects3 for third element, subtract 2/3 width from one,
                # subtract 1/3 width from second, add 1/3 to third (if too far to left, make number being subtracted smaller)
                label='Grover')
rects2 = ax.bar(ind + width/2, davinci_props, width,
                label='GPT-3')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Proportion of Incoherent Relations')
ax.set_title('Proportion of Incoherent Relations Among Grover and GPT-3')
ax.set_xticks(ind)
plt.xticks(rotation=45)
ax.set_xticklabels(['temporal', 'violated\nexpectation', 'cause-effect', 'same', 'contrast', 'similarity',
                    'attribution', 'example', 'condition', 'degenerate', 'generalization'])
ax.legend()

def autolabel(rects, xpos='center'): # creates a graph comparing the average Coh tyoe count between grover
# and human documents

    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


autolabel(rects1, "left")
autolabel(rects2, "right")

fig.tight_layout()

plt.show()

########################################################

