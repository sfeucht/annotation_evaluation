# run to produce graph of average per-document count for each coherence relation type
# (compares grover and human)

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
G_Doc_container, H_SE_container, H_Coh_container, H_Doc_container, D_SE_container, D_Coh_container,
D_Doc_container, SE_accounted_for, Coh_accounted_for, doc_counter)

Coh_types = ['elab', 'temp', 've', 'ce', 'same', 'contr', 'sim', 'attr', 'examp', 'cond', 'deg', 'gen']

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
H_Coh_container = annotator_tag(H_Coh_container)
G_Coh_container = annotator_tag(G_Coh_container)

########################################################
### AVERAGE PER-DOCUMENT COUNT OF EACH COH BETWEEN GROVER AND HUMAN DOCS

def Coh_counts(Doc_container, Coh_container, type): # produces a list with counts of a specific COH type
# from each document, called by avg_count_calculator

    quality_dict = {}

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys():
            for Coh_id in Coh_container[annotator].keys():

                Coh_lists = Coh_container[annotator][doc_id]

                if doc_id == Coh_id: # to match IDs across dicts

                    for list in Coh_lists:

                        if(list[4] == type):
                            if doc_id not in quality_dict: # if doc_id not in quality_dict, creates new entry for it
                                quality_dict[doc_id] = {}
                            if(type not in quality_dict[doc_id]): # adds counter for each COH type for each doc_id
                                quality_dict[doc_id][type] = 0
                            quality_dict[doc_id][type] += 1
                    if doc_id not in quality_dict: # so COH types that don't appear in doc aren't left out of the dict
                        quality_dict[doc_id] = {}
                        quality_dict[doc_id][type] = 0

    return quality_dict

def avg_count_calculator(Doc_container, Coh_container): # produces dictionary with list of the average count and standard
# error per document for each COH type, used in the process of creating bar graph below

        output_dict = {}
        averages = []
        standard_errors = []

        for Coh in Coh_types:
            quality_dict = Coh_counts(Doc_container, Coh_container, Coh)
            quality_list = []

            for k,v in quality_dict.items(): # creates array of counts so that mean, standard error can be calculated for each COH type
                entry = v

                for k1,v1 in entry.items():
                    quality_list.append(v1)

            averages.append(round(mean(quality_list), 2)) # add each mean to list
            standard_errors.append(round(scipy.stats.sem(quality_list), 2)) # add each standard error to list

        output_dict['Average number'] = averages # list of averages from output_dict will be used in creating graph below
        output_dict['Standard errors'] = standard_errors # list of standard errors from output_dict will be used in creating graph below

        return output_dict

# setup for graph function

human_data = avg_count_calculator(H_Doc_container, H_Coh_container)
grover_data = avg_count_calculator(G_Doc_container, G_Coh_container)

human_means, human_sem = human_data['Average number'], human_data['Standard errors']
grover_means, grover_sem = grover_data['Average number'], grover_data['Standard errors']

ind = np.arange(len(human_means))  # the x locations for the groups
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, human_means, width, yerr=human_sem,
                label='Human')
rects2 = ax.bar(ind + width/2, grover_means, width, yerr=grover_sem,
                label='Grover')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Per-Document Count')
ax.set_title('Average Per-Document Count for Each Coherence Relation Type')
ax.set_xticks(ind)
plt.xticks(rotation=45)
ax.set_xticklabels(['elab', 'temporal', 'violated\nexpectation', 'cause-effect', 'same', 'contrast', 'similarity',
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
