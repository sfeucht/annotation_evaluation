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

# helper that either removes or adds an x to a relation's name. 
# e.g. ['56', '57', '59', '59', 'sim'] would become ['56', '57', '59', '59', 'simx']
# e.g. ['5', '5', '9', '9', 'contrx'] would become ['5', '5', '9', '9', 'contr']
# it will add an x to 'deg' or 'rep', even though that's not technically in the annotation schema,
# because it doesn't hurt to have that extra thing to check for anyway. 
def change_x(relation):
    assert(len(relation) == 5)
    if relation[4][-1] == 'x':
        return relation[:4] + [relation[4].strip('x')]
    else:
        return relation[:4] + [relation[4] + 'x']

# helper that takes in a relation and returns list of all the segments in that relation
# i.e. ['0', '4', '5', '5', 'elab'] would become [[0, 1, 2, 3, 4], [5], 'elab']
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
        assert(int(relation[0]) <= int(relation[1]))
        assert(int(relation[2]) <= int(relation[3]))
        beginning_segments = list(range(int(relation[0]), int(relation[1])+1))
        end_segments = list(range(int(relation[2]), int(relation[3])+1))
    
    return [beginning_segments, end_segments, relation[4]]

# helper that takes two unrolled relations and returns 
#   True if both beginning and end boundaries overlap
#   False otherwise 
def boundaries_overlap(this_unrolled, that_unrolled):
    assert(len(this_unrolled) == len(that_unrolled) == 3)
    # check if beginning overlaps 
    beginning_overlaps = False
    for n in this_unrolled[0]:
        if n in that_unrolled[0]: 
            beginning_overlaps = True

    # check if end overlaps, only bother checking if beginning overlaps too 
    end_overlaps = False
    if beginning_overlaps:
        for m in this_unrolled[1]:
            if m in that_unrolled[1]:
                end_overlaps = True
    
    return (beginning_overlaps and end_overlaps)


# function that takes in two coherence relation containers, returns agreement kappa score
def coherence_agreement(larger, smaller):
    print(len(larger), len(smaller))
    larger_original_len = len(larger)
    smaller_original_len = len(smaller)

    for this_relation in larger:

        # see if there's an exact match, if so remove it from both docs
        if this_relation in smaller:
            smaller.remove(this_relation)
            larger.remove(this_relation)

        # see if there's an exact match but with an extra/removed x as well
        elif change_x(this_relation) in smaller:
            smaller.remove(change_x(this_relation))
            larger.remove(this_relation)

        # see if there's an exact flipped match, if so remove from both
        elif flip(this_relation) in smaller:
            smaller.remove(flip(this_relation))
            larger.remove(this_relation)

        # see if there's an exact flipped match but with an extra/removed x 
        elif flip(change_x(this_relation)) in smaller:
            smaller.remove(flip(change_x(this_relation)))
            larger.remove(this_relation)

    
        else: 
            # then, compare to see if unrolled versions are essentially the same
            # slightly different boundaries but the same annotation
            this_unrolled = unroll(this_relation)

            # go though all the relations in smaller and find an overlapping one
            for that_relation in smaller:
                that_unrolled = unroll(that_relation)

                # if they are the same relation label, then compare boundaries
                if this_unrolled[2] == that_unrolled[2]: #TODO add toggling for same samex
                    # then remove first occurrence if the boundaries do overlap 
                    if boundaries_overlap(this_unrolled, that_unrolled):
                        smaller.remove(that_relation)
                        larger.remove(this_relation)
                        break
                    else: 
                        # if this is a symmetrical relation, check again but with
                        # a flipped version of this_relation comparing to that_relation.
                        # flip() only changes this_relation if it's symmetrical.
                        if this_relation != flip(this_relation): 
                            if boundaries_overlap(unroll(flip(this_relation)), that_unrolled):
                                smaller.remove(that_relation)
                                larger.remove(this_relation)
                                break

    
    print(len(larger), len(smaller))

    # looking at leftovers, create two same-length vectors that can be used to calculate kappa
    # first get how many relations were removed from both docs, which is the number that matched
    assert(larger_original_len - len(larger) == smaller_original_len - len(smaller))
    number_matching = larger_original_len - len(larger)

    # get the two vector representations: the leftovers in larger and smaller,
    # plus the number of relations that actually matched between the two
    new_vector_length = number_matching + len(larger) + len(smaller)
    larger_vector = [2]*number_matching + [1]*len(larger) + [0]*len(smaller)
    smaller_vector = [2]*number_matching + [0]*len(larger) + [1]*len(smaller)

    assert(len(larger_vector) == len(smaller_vector) == new_vector_length)
    return cohen_kappa_score(larger_vector, smaller_vector)



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
            score = coherence_agreement(a_container, b_container)
            print(doc_id, score, len(a_container), len(b_container))
        else:
            score = coherence_agreement(b_container, a_container)
            print(doc_id, score, len(a_container), len(b_container))
