import random
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from extract_annotations import fill_in_human_grover, fill_in_containers, fill_in_SE_robust


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

# Get robust version of SE containers
G_SE_container_robust = {"Sheridan":{},"Muskaan":{},"Kate":{}}
H_SE_container_robust = {"Sheridan":{},"Muskaan":{},"Kate":{}}
SE_accounted_for_2 = []
doc_counter_2 = 0

doc_counter_2 = fill_in_SE_robust(h_docs, g_docs, G_SE_container_robust, H_SE_container_robust, SE_accounted_for_2, doc_counter_2)

# Agreement for SE types
kappa_list = []
agreement_by_pair = {
    'Sheridan and Muskaan': [], 
    'Sheridan and Kate': [], 
    'Muskaan and Kate': []
}

disagreement_dict = {}
top_10_combinations = [ # from running this code before
    'BASIC STATE GENERIC SENTENCE (STATIC)', 
    'GENERIC SENTENCE (DYNAMIC) UNBOUNDED EVENT (GENERIC)', 
    'OTHER GENERIC SENTENCE (STATIC)', 
    'GENERIC SENTENCE (STATIC) COERCED STATE (GENERIC)', 
    'GENERALIZING SENTENCE (DYNAMIC) GENERIC SENTENCE (STATIC)',
    'UNBOUNDED EVENT (SPECIFIC) BASIC STATE',
    'COERCED STATE (GENERIC) COERCED STATE (SPECIFIC)',
    'GENERIC SENTENCE (STATIC) GENERIC SENTENCE (DYNAMIC)',
    'COERCED STATE (GENERIC) UNBOUNDED EVENT (GENERIC)',
    'COERCED STATE (SPECIFIC) BASIC STATE'
] 
top_10_combinations_dict = {key:[] for key in top_10_combinations}

for doc_id in h_docs + g_docs:
    tuples = [t for t in SE_accounted_for if t[0]==doc_id]
    if len(tuples) > 1: 
        assert(len(tuples)==2)
        _, a_annotator, is_human = tuples[0]
        _, b_annotator, _ = tuples[1]

        if is_human:
            a_container = H_SE_container[a_annotator][doc_id] 
            b_container = H_SE_container[b_annotator][doc_id]
            a_container_robust = H_SE_container_robust[a_annotator][doc_id]
            b_container_robust = H_SE_container_robust[b_annotator][doc_id]
        else:
            a_container = G_SE_container[a_annotator][doc_id] 
            b_container = G_SE_container[b_annotator][doc_id]
            a_container_robust = G_SE_container_robust[a_annotator][doc_id]
            b_container_robust = G_SE_container_robust[b_annotator][doc_id]

        # get and save kappa score for these two docs
        assert(len(a_container) == len(b_container))
        score = cohen_kappa_score(a_container, b_container)
        kappa_list += [[doc_id, 'human' if is_human else 'grover', score, a_annotator, b_annotator]]

        # also see what kappa is if there's two long vectors for each pair of annotators
        if (a_annotator == 'Sheridan' and b_annotator == 'Muskaan') or (b_annotator == 'Sheridan' and a_annotator == 'Muskaan'):
            pair = agreement_by_pair['Sheridan and Muskaan']
        elif (a_annotator == 'Sheridan' and b_annotator == 'Kate') or (b_annotator == 'Sheridan' and a_annotator == 'Kate'):
            pair = agreement_by_pair['Sheridan and Kate']
        elif (a_annotator == 'Muskaan' and b_annotator == 'Kate') or (b_annotator == 'Muskaan' and a_annotator == 'Kate'):
            pair = agreement_by_pair['Muskaan and Kate']
        pair.append(a_container)
        pair.append(b_container)

        # store counts of different types of disagreements in dictionary
        for a, b in zip(a_container_robust, b_container_robust):
            if a[0] != b[0]:
                # store counts in disagreement_dict 
                if (a[0] + ' ' + b[0]) in disagreement_dict.keys():
                    disagreement_dict[a[0] + ' ' + b[0]] += 1
                    key = a[0] + ' ' + b[0]
                elif (b[0] + ' ' + a[0]) in disagreement_dict.keys():
                    disagreement_dict[b[0] + ' ' + a[0]] += 1
                    key = b[0] + ' ' + a[0]
                else:
                    disagreement_dict[a[0] + ' ' + b[0]] = 1
                    key = a[0] + ' ' + b[0]
                
                # if it's one of the top 10 disagreements, dump in disagreement_full_dict
                if key in top_10_combinations_dict.keys():
                    top_10_combinations_dict[key] += [[doc_id, a_annotator, a[0], b_annotator, b[0], b[1]]]




kappa_scores = pd.DataFrame(kappa_list, columns=['doc_id', 'type', 'cohen_kappa', 'a_annotator', 'b_annotator'])
print(kappa_scores.sort_values('cohen_kappa'))
print("overall mean kappa score: ", kappa_scores['cohen_kappa'].mean())
print("human mean kappa score: ", kappa_scores[(kappa_scores['type'] == 'human')]['cohen_kappa'].mean())
print("grover mean kappa score: ", kappa_scores[(kappa_scores['type'] == 'grover')]['cohen_kappa'].mean())

print("Sheridan and Muskaan: ", kappa_scores[(kappa_scores['a_annotator'] == 'Sheridan') & (kappa_scores['b_annotator'] == 'Muskaan')]['cohen_kappa'].mean())
print("Muskaan and Kate: ", kappa_scores[(kappa_scores['a_annotator'] == 'Muskaan') & (kappa_scores['b_annotator'] == 'Kate')]['cohen_kappa'].mean())
print("Sheridan and Kate: ", kappa_scores[(kappa_scores['a_annotator'] == 'Sheridan') & (kappa_scores['b_annotator'] == 'Kate')]['cohen_kappa'].mean())

print('\n' + 'agreement concatenating docs together:')
for k in agreement_by_pair.keys():
    print(k, cohen_kappa_score(agreement_by_pair[k][0], agreement_by_pair[k][1]))


'''
disagreement_df = pd.DataFrame.from_dict({'combination' : disagreement_dict.keys(), 'count' : disagreement_dict.values()})
pd.set_option('display.max_colwidth', None)
print(disagreement_df.sort_values('count', ascending=False).head(30))

# randomly choose 10 of each type of combination from top_10_combinations_dict
for key in top_10_combinations_dict.keys():
    short_list = []

    while len(short_list) < 10:
        i = random.randrange(len(top_10_combinations_dict[key]))
        short_list.append(top_10_combinations_dict[key][i])
        del top_10_combinations_dict[key][i]

    df = pd.DataFrame(short_list, columns=['doc_id', 'a_annotator', 'a_label', 'b_annotator', 'b_label', 'text'])
    df.to_csv('100_se_disagreements/' + key + '.csv')
'''