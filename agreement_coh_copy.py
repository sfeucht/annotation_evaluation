# this file is a copy of agreement_coh.py, and is testing to see what agreement looks like with different changes.
# DIFFERENT IN THIS COPY: 
#   - boundaries_overlap needs only beginning or end to overlap
import random
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
# here, we decided to consider elab symmetrical, even though theoretically they should not be.
def flip(relation):
    assert(len(relation) == 5)
    if relation[4] in ['same', 'samex', 'deg', 'sim', 'simx', 'contr', 'contrx', 'rep', 'elab']:
        relation_copy = relation[:]
        second_two = relation[2:4]
        del relation_copy[2:4]
        return second_two + relation_copy
    else:
        return relation

# helper that switches around the line numbers, regardless of whether it's symmetrical or not
def flip_hard(relation):
    assert(len(relation) == 5)
    relation_copy = relation[:]
    second_two = relation[2:4]
    del relation_copy[2:4]
    return second_two + relation_copy

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
#   True if the beginnings overlap, or if the ends overlap 
#   True if beginning of one overlaps end of the other, or end of one overlaps beginning of other
#   False otherwise 
def boundaries_overlap(this_unrolled, that_unrolled):
    assert(len(this_unrolled) == len(that_unrolled) == 3)
    opposites_overlap = False
    
    # check if beginning overlaps 
    beginning_overlaps = False
    for n in this_unrolled[0]:
        if n in that_unrolled[0]: 
            beginning_overlaps = True
        if n in that_unrolled[1]:
            opposites_overlap = True

    # check if end overlaps
    end_overlaps = False
    for m in this_unrolled[1]:
        if m in that_unrolled[1]:
            end_overlaps = True
        if m in that_unrolled[0]:
            opposites_overlap = True
    
    return (beginning_overlaps or end_overlaps or opposites_overlap)

# helper that checks if two labels are equal 
# considers 'same' and 'elab' equal, 'cex' and 'ce' equal 
def labels_equal(label1, label2):
    if (label1 == 'same' and label2 in ['elab', 'elabx']) or (label2 == 'same' and label1 in ['elab', 'elabx']):
        return True
    elif label1[-1] == 'x' and label2[-1] != 'x':
        return label1[:-1] == label2
    elif label1[-1] != 'x' and label2[-1] == 'x':
        return label1 == label2[:-1]
    else:
        return label1 == label2
    

# for testing purposes, removes all elab and elabxes from a given SE container 
def remove_elabs(container):
    for relation in container:
        assert(len(relation) == 5)
        if relation[4] == 'elab' or relation[4] == 'elabx':
            container.remove(relation)

# for testing purposes, removes all incoherent relations from given SE container,
# returns container with just the incoherent relations
def remove_incoherent(container):
    incoherent = []
    for relation in container:
        assert(len(relation) == 5)
        if relation[4][-1] == 'x' or relation in ['deg', 'rep']:
            container.remove(relation)
            incoherent.append(relation)
    return incoherent


# function that takes in two coherence relation containers, returns agreement kappa score
def coherence_agreement(larger, smaller):
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
                if labels_equal(this_unrolled[2], that_unrolled[2]):
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


    # looking at leftovers, create two same-length vectors that can be used to calculate kappa.
    # first get how many relations were removed from both docs, which is the number that matched
    assert(larger_original_len - len(larger) == smaller_original_len - len(smaller))
    number_matching = larger_original_len - len(larger)

    # get the two vector representations: the leftovers in larger and smaller,
    # plus the number of relations that actually matched between the two
    new_vector_length = number_matching + len(larger) + len(smaller)
    larger_vector = [2]*number_matching + [1]*len(larger) + [0]*len(smaller)
    smaller_vector = [2]*number_matching + [0]*len(larger) + [1]*len(smaller)

    assert(len(larger_vector) == len(smaller_vector) == new_vector_length)
    return (cohen_kappa_score(larger_vector, smaller_vector), number_matching)


def most_common_disagreements(larger, l_annotator, smaller, s_annotator, doc_id):
    for this_relation in larger:
        # then, compare to see if unrolled versions are essentially the same
        # slightly different boundaries but the same annotation
        this_unrolled = unroll(this_relation)
        this_x_unrolled = unroll(change_x(this_relation))

        # go though all the relations in smaller and find an overlapping one
        for that_relation in smaller:
            that_unrolled = unroll(that_relation)

            # check whether they overlap 
            if boundaries_overlap(this_unrolled, that_unrolled) or boundaries_overlap(unroll(flip_hard(this_relation)), that_unrolled):
                # if the labels don't match, record in disagreement dict 
                if this_unrolled[2] != that_unrolled[2] and this_x_unrolled[2] != that_unrolled[2]:
                    a = change_x(this_relation)[4] if this_unrolled[2][-1]=='x' else this_unrolled[2]
                    b = change_x(that_relation)[4] if that_unrolled[2][-1]=='x' else that_unrolled[2]

                    if (a + ' ' + b) in disagreement_dict.keys():
                        disagreement_dict[a + ' ' + b] += 1
                        key = a + ' ' + b
                    elif (b + ' ' + a) in disagreement_dict.keys():
                        disagreement_dict[b + ' ' + a] += 1
                        key = b + ' ' + a
                    else:
                        disagreement_dict[a + ' ' + b] = 1
                        key = a + ' ' + b
                    
                    # if it's one of the top 10 disagreements, dump in disagreement_full_dict
                    if key in top_10_combinations_dict.keys():
                        top_10_combinations_dict[key] += [[doc_id, l_annotator, this_relation, s_annotator, that_relation]]

kappa_list = []
disagreement_dict = {}
top_10_combinations = [ # from running this code before
    'ce elab',
    'elab sim',
    'elab ve',
    'elab attr',
    'cond elab',
    'elab temp',
    'examp elab',
    'elab deg',
    'ce sim',
    've contr'
]
top_10_combinations_dict = {key:[] for key in top_10_combinations}

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

        # remove all elab and elabxes from containers
        # remove_elabs(a_container)
        # remove_elabs(b_container)

        # remove incoherent relations and place them into container by themselves
        # a_container = remove_incoherent(a_container)
        # b_container = remove_incoherent(b_container)

        if len(a_container) >= len(b_container):
            score, number_matching = coherence_agreement(a_container, b_container)
            most_common_disagreements(a_container, a_annotator, b_container, b_annotator, doc_id)
        else:
            score, number_matching = coherence_agreement(b_container, a_container)
            most_common_disagreements(b_container, b_annotator, a_container, a_annotator, doc_id)
        
        kappa_list += [[doc_id, 'human' if is_human else 'grover', score, a_annotator, b_annotator, len(a_container), len(b_container), number_matching]]


kappa_scores = pd.DataFrame(kappa_list, columns=['doc_id', 'type', 'cohen_kappa', 'a_annotator', 'b_annotator', 'a_no_annotations', 'b_no_annotations', 'number_matching'])
#print(kappa_scores.sort_values('cohen_kappa'))
print(kappa_scores)

print("overall mean kappa score: ", kappa_scores['cohen_kappa'].mean())
print("human mean kappa score: ", kappa_scores[(kappa_scores['type'] == 'human')]['cohen_kappa'].mean())
print("grover mean kappa score: ", kappa_scores[(kappa_scores['type'] == 'grover')]['cohen_kappa'].mean())

print("Sheridan and Muskaan: ", kappa_scores[(kappa_scores['a_annotator'] == 'Sheridan') & (kappa_scores['b_annotator'] == 'Muskaan')]['cohen_kappa'].mean())
print("Muskaan and Kate: ", kappa_scores[(kappa_scores['a_annotator'] == 'Muskaan') & (kappa_scores['b_annotator'] == 'Kate')]['cohen_kappa'].mean())
print("Sheridan and Kate: ", kappa_scores[(kappa_scores['a_annotator'] == 'Sheridan') & (kappa_scores['b_annotator'] == 'Kate')]['cohen_kappa'].mean())

'''
disagreement_df = pd.DataFrame.from_dict({'combination' : disagreement_dict.keys(), 'count' : disagreement_dict.values()})
pd.set_option('display.max_colwidth', None)
print(disagreement_df.sort_values('count', ascending=False).head(10))

# randomly choose 10 of each type of combination from top_10_combinations_dict
for key in top_10_combinations_dict.keys():
    lst = top_10_combinations_dict[key]
    #sample = random.sample(lst, min(10, len(lst)))
    sample = lst #actually, let's just take the whole list

    df = pd.DataFrame(sample, columns=['doc_id', 'a_annotator', 'a_relation', 'b_annotator', 'b_relation'])
    df.to_csv('100_coh_disagreements/' + key + '.csv')
'''


'''
incoherences_marked = kappa_scores[(kappa_scores['a_no_annotations'] > 0) | (kappa_scores['b_no_annotations'] > 0)]
print(incoherences_marked)
print("mean kappa score for docs with incoherences marked: ", incoherences_marked['cohen_kappa'].mean())
print("human mean kappa score: ", incoherences_marked[(incoherences_marked['type'] == 'human')]['cohen_kappa'].mean())
print("grover mean kappa score: ", incoherences_marked[(incoherences_marked['type'] == 'grover')]['cohen_kappa'].mean())

print("Sheridan and Muskaan: ", incoherences_marked[(incoherences_marked['a_annotator'] == 'Sheridan') & (incoherences_marked['b_annotator'] == 'Muskaan')]['cohen_kappa'].mean())
print("Muskaan and Kate: ", incoherences_marked[(incoherences_marked['a_annotator'] == 'Muskaan') & (incoherences_marked['b_annotator'] == 'Kate')]['cohen_kappa'].mean())
print("Sheridan and Kate: ", incoherences_marked[(incoherences_marked['a_annotator'] == 'Sheridan') & (incoherences_marked['b_annotator'] == 'Kate')]['cohen_kappa'].mean())
'''


