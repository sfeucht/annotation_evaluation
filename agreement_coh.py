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


# Agreement for Coherence relations

# helper that switches around the line numbers. only does this if the relation is symmetrical 
def flip(relation):
    assert(len(relation) == 5)
    if relation[4] in ['same', 'samex', 'deg', 'sim', 'simx', 'contr', 'contrx', 'rep']:
        relation_copy = relation[:]
        second_two = relation[2:4]
        del relation_copy[2:4]
        return second_two + relation_copy
    else:
        return relation

# TODO: helper that takes in a relation and returns list of all the segments in that relation
# i.e. ['0', '4', '5', '5', 'elab'] would become [['0', '1', '2', '3', '4'], ['5'], 'elab']
def unroll(relation):
    assert(len(relation) == 5)
    if '?' == relation[0] or '?' == relation[1]:
        beginning_segments = ['?']
        end_segments = list(range(int(relation[2]), int(relation[3])+1))
    elif '?' == relation[2] or '?' == relation[3]:
        beginning_segments = list(range(int(relation[0]), int(relation[1])+1))
        end_segments = ['?']
    else: # else, the first four should be numbers
        assert('?' not in relation)
        beginning_segments = list(range(int(relation[0]), int(relation[1])+1))
        end_segments = list(range(int(relation[2]), int(relation[3])+1))
    
    return [beginning_segments, end_segments, relation[4]]
    


# TODO: function that takes in two coherence relation containers, returns ______
def coherence_agreement(larger, smaller):
    matching = 0
    for this_relation in larger:
        # if matching is incremented at any point, turn to True, so we know to remove
        accounted_for = False

        # see if there's an exact match, if so remove it from smaller, but check other cases still
        if this_relation in smaller:
            #smaller.remove(this_relation)
            accounted_for = True
            matching += 1
        
        # see if there's an exact flipped match, if so remove from smaller, but check other cases still
        if flip(this_relation) in smaller:
            #smaller.remove(flip(this_relation))
            accounted_for = True
            matching += 1
    
        # then, compare to see if unrolled versions are essentially the same
        # slightly different boundaries but the same annotation
        this_unrolled = unroll(this_relation)

        # go though all the relations in smaller and count any overlapping ones
        for that_relation in smaller:
            that_unrolled = unroll(that_relation)

            # if they are the same relation
            if this_relation[2] == that_unrolled[2]:
                for n in this_unrolled[0]:
                    if n in that_unrolled[0]: 
                        #if there's an overlap in beginning, check end. 
                        for m in this_unrolled[1]:
                            if m in that_unrolled[1]:
                                #smaller.remove(that_relation)
                                accounted_for = True
                                matching += 1

        # if this is a symmetrical relation, do the same overlap checking for everything in smaller, but flipped
        if this_relation != flip(this_relation):
            this_flipped_unrolled = unroll(flip(this_relation))
            for that_relation in smaller:
                that_unrolled = unroll(that_relation)
                
                # if they are the same relation
                if this_flipped_unrolled[2] == that_unrolled[2]:
                    for n in this_flipped_unrolled[0]:
                        if n in that_unrolled[0]: 
                            #if there's an overlap in beginning, check end. 
                            for m in this_flipped_unrolled[1]:
                                if m in that_unrolled[1]:
                                    #smaller.remove(that_relation)
                                    accounted_for = True
                                    matching += 1

        # if matching was incremented at any point, then remove this_relation from larger
        if accounted_for:
            pass
            #larger.remove(this_relation)
    
    # TODO: after going through all the relations in larger, count leftovers in both. 
    return matching


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
            matching = coherence_agreement(a_container, b_container)
            print(doc_id, matching, len(a_container), len(b_container))
        else:
            matching = coherence_agreement(b_container, a_container)
            print(doc_id, matching, len(a_container), len(b_container))
