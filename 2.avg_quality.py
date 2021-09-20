# run to produce graph of average quality measures (grover vs human)

import csv
import re
import os
import sys

import scipy
import pandas as pd
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
            'QUESTION', 'OTHER']

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
H_SE_container = annotator_tag(H_SE_container)
G_SE_container = annotator_tag(G_SE_container)
D_SE_container = annotator_tag(D_SE_container)

########################################################
### AVERAGE QUALITY MEASURES BETWEEN GROVER AND HUMAN DOCS

def avg_quality_measures(Doc_container): # produces dictionary with list of the average quality measures and standard
# errors across all documents, used in the process of creating bar graph below

    indexes = [4, 5, 6, 7, 13, 14]

    output_dict = {}
    averages = []
    standard_errors = []

    for index in indexes: # calls the index for each of the quality measures

        rating_list = []

        for annotator in Doc_container.keys():
            for doc_id in Doc_container[annotator].keys():
                ratings = Doc_container[annotator][doc_id]

                if ratings[index] != 'NA' and ratings[index] != '0': # if a rating exists for this index, add rating to list
                    rating_list.append(int(ratings[index]))

        averages.append(round(mean(rating_list), 2)) # take the list of quality ratings above and append its average to this list
        standard_errors.append(round(scipy.stats.sem(rating_list), 2)) # same for standard error

    output_dict['Average quality'] = averages # list of averages from output_dict will be used in creating graph below
    output_dict['Standard errors'] = standard_errors # list of standard errors from output_dict will be used in creating graph below

    return output_dict

# setup for graph function

human_data = avg_quality_measures(H_Doc_container)
grover_data = avg_quality_measures(G_Doc_container)

human_means, human_sem = human_data['Average quality'], human_data['Standard errors']
grover_means, grover_sem = grover_data['Average quality'], grover_data['Standard errors']

ind = np.arange(len(human_means))  # the x locations for the groups
width = 0.40  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, human_means, width, yerr=human_sem,
                label='Human')
rects2 = ax.bar(ind + width/2, grover_means, width, yerr=grover_sem,
                label='Grover')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Quality Rating (on 1-5 scale)')
ax.set_title('Average Quality Rating by Document Type')
ax.set_xticks(ind)
plt.xticks(rotation=45)
ax.set_xticklabels(['Plausibility (N)', 'Completeness (N)', 'Consistency (N)', 'Coverage (N)', 'Cogency (A)',
                    'Effectiveness (A)'])
ax.legend()

def autolabel(rects, xpos='center'): # creates a graph comparing the average quality ratings between grover
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
########################################################
