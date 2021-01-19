import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from extract_annotations import fill_in_human_grover, fill_in_containers


# First, extract all of the SE types and coh relations and put into containers.

# get which docs are human and not
h_docs = []
g_docs = []
fill_in_human_grover(h_docs, g_docs)

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


# Agreement for SE types
kappa_list = []
for doc_id in h_docs + g_docs:
    tuples = [t for t in SE_accounted_for if t[0]==doc_id]
    if len(tuples) > 1: 
        assert(len(tuples)==2)
        _, a_annotator, is_human = tuples[0]
        _, b_annotator, _ = tuples[1]

        if is_human:
            a_container = H_SE_container[a_annotator][doc_id] 
            b_container = H_SE_container[b_annotator][doc_id]
        else:
            a_container = G_SE_container[a_annotator][doc_id] 
            b_container = G_SE_container[b_annotator][doc_id]

        # if len(a_container) != len(b_container):
        #     print(doc_id, len(a_container), len(b_container), a_annotator, b_annotator)

        assert(len(a_container) == len(b_container))
        score = cohen_kappa_score(a_container, b_container)

        kappa_list += [[doc_id, 'human' if is_human else 'grover', score, a_annotator, b_annotator]]


kappa_scores = pd.DataFrame(kappa_list, columns=['doc_id', 'type', 'cohen_kappa', 'a_annotator', 'b_annotator'])
print(kappa_scores.sort_values('cohen_kappa'))
print("overall mean kappa score: ", kappa_scores['cohen_kappa'].mean())
print("human mean kappa score: ", kappa_scores[(kappa_scores['type'] == 'human')]['cohen_kappa'].mean())
print("grover mean kappa score: ", kappa_scores[(kappa_scores['type'] == 'grover')]['cohen_kappa'].mean())

print("Sheridan and Muskaan: ", kappa_scores[(kappa_scores['a_annotator'] == 'Sheridan') & (kappa_scores['b_annotator'] == 'Muskaan')]['cohen_kappa'].mean())
print("Muskaan and Kate: ", kappa_scores[(kappa_scores['a_annotator'] == 'Muskaan') & (kappa_scores['b_annotator'] == 'Kate')]['cohen_kappa'].mean())
print("Sheridan and Kate: ", kappa_scores[(kappa_scores['a_annotator'] == 'Sheridan') & (kappa_scores['b_annotator'] == 'Kate')]['cohen_kappa'].mean())

