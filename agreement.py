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

'''
# Agreement for SE types
kappa_scores = []
for doc_id in h_docs + g_docs:
    tuples = [t for t in SE_accounted_for if t[0]==doc_id]
    if len(tuples) > 1: 
        print(doc_id)
        assert(len(tuples)==2)
        _, a_annotator, is_human = tuples[0]
        _, b_annotator, _ = tuples[1]

        if is_human:
            a_container = H_SE_container[a_annotator][doc_id] 
            b_container = H_SE_container[b_annotator][doc_id]
        else:
            a_container = G_SE_container[a_annotator][doc_id] 
            b_container = G_SE_container[b_annotator][doc_id]

        assert(len(a_container) == len(b_container))

        kappa_scores += (doc_id, cohen_kappa_score(a_container, b_container), a_annotator, b_annotator)
'''

# Agreement for Coherence relations

# helper that switches around the line numbers. only does this if the relation is symmetrical 
def flip(relation):
    assert(len(relation) == 5)
    if relation[4] in ['same', 'samex', 'deg', 'sim', 'simx', 'contr', 'contrx', 'rep']:
        second_two = relation[2:4]
        del relation[2:4]
        return second_two + relation
    else:
        return relation

# TODO: helper that takes in a relation and returns list of all the segments in that relation
# i.e. ['0', '4', '5', '5', 'elab'] would become [['0', '1', '2', '3', '4'], ['5'], 'elab']
def unroll(relation):
    return relation

# TODO: function that takes in two coherence relation containers, returns ______
def coherence_agreement(larger, smaller):
    matching = 0
    for relation in larger:
        if relation in smaller or flip(relation) in smaller:
            larger.remove(relation)
            matching += 1
        else:
            unrolled = unroll(relation)
            # TODO: comparing slightly different boundaries but same annotation


for doc_id in h_docs + g_docs:
    tuples = [t for t in Coh_accounted_for if t[0]==doc_id]
    if len(tuples) > 1: 
        assert(len(tuples)==2)
        _, a_annotator, is_human = tuples[0]
        _, b_annotator, _ = tuples[1]

        if is_human:
            a_container = H_Coh_container[a_annotator][doc_id] 
            b_container = H_Coh_container[b_annotator][doc_id]
        else:
            a_container = G_Coh_container[a_annotator][doc_id] 
            b_container = G_Coh_container[b_annotator][doc_id]

        if len(a_container) >= len(b_container):
            coherence_agreement(a_container, b_container)
        else:
            coherence_agreement(b_container, a_container)