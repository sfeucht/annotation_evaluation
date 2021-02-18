import random
import numpy as np
import pandas as pd
import krippendorff_alpha as ka
from extract_annotations import fill_in_human_grover, fill_in_containers
from copy import deepcopy

# all the top_10 stuff commented out is for sampling actual examples of disagreeing relations from the documents
# these are macros for the whole file so you can change things easily 
boundaries_lenient = False
ignore_elab_disagreements = True

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


# helper that takes two unrolled relations, and 
# if boundaries_lenient=False, returns True only if beginnings and ends both overlap
# if boundaries_lenient=True, returns True if the beginnings or the ends overlap 
def boundaries_overlap(this_unrolled, that_unrolled):
    assert(len(this_unrolled) == len(that_unrolled) == 3)
    
    # check if beginning overlaps 
    beginning_overlaps = False
    for n in this_unrolled[0]:
        if n in that_unrolled[0]: 
            beginning_overlaps = True

    # check if end overlaps
    end_overlaps = False
    for m in this_unrolled[1]:
        if m in that_unrolled[1]:
            end_overlaps = True
    
    if not boundaries_lenient:
        return beginning_overlaps and end_overlaps
    elif boundaries_lenient:
        return beginning_overlaps or end_overlaps

# helper that checks if two labels are equal 
# considers these cases: 
#   'cex' and 'ce' equal 
#   'same' and 'elab' equal
#   'attr' and 'attrm' equal attr
#   if the macro ignore_elab_disagreements is True, consider elab equal to any other relation.
def labels_equal(label1, label2):
    if ignore_elab_disagreements and any(l in ['elab', 'elabx'] for l in [label1, label2]): 
        return True # if either of the labels are elab and macro on, mark as agreement
    elif (label1 == 'same' and label2[:4] == 'elab') or (label2 == 'same' and label1[:4] == 'elab'):
        return True
    elif label1[:4] == 'attr' and label2[:4] == 'attr':
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

# a class to hold the two dictionaries and counts of matches for a pair of documents
class DocPairCoherences:
    def __init__(self):
        self.number_overlapping = 0 # number of relations that overlap each other 
        self.number_matching = 0 # number of relations that match labels and boundaries
        self.dict_tab = 0 # to keep track of how many things in each dict
        self.larger_dict = {}
        self.smaller_dict = {}
        self.larger_annotator = ''
        self.smaller_annotator = ''

    def update_dicts(self, l, s):
        self.dict_tab += 1 
        self.larger_dict[self.dict_tab] = l
        self.smaller_dict[self.dict_tab] = s
    
    def increment_number_overlapping(self, labels_match):
        self.number_overlapping += 1
        if labels_match:
            self.number_matching += 1
    
    def set_which_annotator(self, larger_annotator, smaller_annotator):
        self.larger_annotator = larger_annotator
        self.smaller_annotator = smaller_annotator

    def get_dict_by_annotator(self, annotator):
        if annotator == self.larger_annotator:
            return self.larger_dict
        elif annotator == self.smaller_annotator:
            return self.smaller_dict
        else:
            Exception("annotator not found")



# function that takes in two coherence relation containers, returns agreement alpha score
def coherence_agreement(larger, smaller):
    larger_original_len = len(larger)
    smaller_original_len = len(smaller)

    # create object to store dictionaries and counts for all the dis/agreements
    pair = DocPairCoherences()

    # iterate over these copies, but remove and check from the originals 
    larger_copy = deepcopy(larger)
    smaller_copy = deepcopy(smaller)

    for this_relation in larger_copy:
        # see if there's an exact match, if so remove it from both docs
        # also add to the appropriate dicts with same label 
        # also increment number_matching and number_overlapping
        if this_relation in smaller:
            pair.increment_number_overlapping(labels_match=True)
            pair.update_dicts(1, 1)
            larger.remove(this_relation)
            smaller.remove(this_relation)

        # see if there's an exact match but with an extra/removed x as well
        elif change_x(this_relation) in smaller:
            pair.increment_number_overlapping(labels_match=True)
            pair.update_dicts(1, 1)
            larger.remove(this_relation)
            smaller.remove(change_x(this_relation))

        # see if there's an exact flipped match, if so remove from both
        elif flip(this_relation) in smaller:
            pair.increment_number_overlapping(labels_match=True)
            pair.update_dicts(1, 1)
            larger.remove(this_relation)
            smaller.remove(flip(this_relation))
            
        # see if there's an exact flipped match but with an extra/removed x 
        elif flip(change_x(this_relation)) in smaller:
            pair.increment_number_overlapping(labels_match=True)
            pair.update_dicts(1, 1)
            larger.remove(this_relation)
            smaller.remove(flip(change_x(this_relation)))
            
        else: 
            # then, compare to see if unrolled versions are essentially the same
            # slightly different boundaries but the same annotation
            this_unrolled = unroll(this_relation)

            # go though all the relations in smaller and find a matching one 
            for that_relation in smaller_copy:
                # we have to use a copy so the loops aren't messed up, but actually
                # we only want to compare if that_relation hasn't been removed yet.
                if that_relation in smaller:
                    that_unrolled = unroll(that_relation)

                    # if they are the same relation label, then compare boundaries
                    if labels_equal(this_unrolled[2], that_unrolled[2]):
                        # then remove first occurrence if the boundaries do overlap 
                        if boundaries_overlap(this_unrolled, that_unrolled):
                            pair.increment_number_overlapping(labels_match=True)
                            pair.update_dicts(1, 1)
                            larger.remove(this_relation)
                            smaller.remove(that_relation)
                            break
                        else: 
                            # if this is a symmetrical relation, check again but with
                            # a flipped version of this_relation comparing to that_relation.
                            # flip() only changes this_relation if it's symmetrical.
                            if this_relation != flip(this_relation): 
                                if boundaries_overlap(unroll(flip(this_relation)), that_unrolled):
                                    pair.increment_number_overlapping(labels_match=True)
                                    pair.update_dicts(1, 1)
                                    larger.remove(this_relation)
                                    smaller.remove(that_relation)
                                    break

    # after going through and noting all of the agreements, do a second pass to match up disagreements.
    #    1. look for exact disagreements and remove those,
    #    2. then look for approximate disagreements, taking the first overlapping one
    for this_relation in larger_copy:
        if this_relation in larger: #again, workaround bc we can't directly iterate thru larger
            for that_relation in smaller_copy:
                if that_relation in smaller:
                    #remove exact disagreements
                    if this_relation[:4] == that_relation[:4]:
                        assert(this_relation != that_relation)
                        pair.increment_number_overlapping(labels_match=False)
                        pair.update_dicts(2, 3)
                        larger.remove(this_relation)
                        smaller.remove(that_relation)
                        break #stop looking through smaller

                    #see if there is an approximate disagreement, if so take it
                    that_unrolled = unroll(that_relation)
                    if boundaries_overlap(this_unrolled, that_unrolled):
                        pair.increment_number_overlapping(labels_match=False)
                        pair.update_dicts(2, 3)
                        larger.remove(this_relation)
                        smaller.remove(that_relation)
                        break
                    #if one of them is flippable, see if there is an overlap when you flip one of them
                    elif (this_relation != flip(this_relation)) or (that_relation != flip(that_relation)):
                        if boundaries_overlap(unroll(flip(this_relation)), that_unrolled):
                            pair.increment_number_overlapping(labels_match=False)
                            pair.update_dicts(2, 3)
                            larger.remove(this_relation)
                            smaller.remove(that_relation)
                            break 
                

    # first get how many relations were removed from both docs == number_matching + number of overlapping not matching
    assert(larger_original_len - len(larger) == smaller_original_len - len(smaller))
    number_removed = larger_original_len - len(larger)

    leftover_start_point = pair.dict_tab + 1
    # add all the leftover ones in larger to larger_dict with unique IDs to count as disagreement
    for i in range(leftover_start_point, leftover_start_point + len(larger)):
        assert(i not in pair.larger_dict.keys())
        assert(i not in pair.smaller_dict.keys())
        pair.larger_dict[i] = 4

    # add all the leftover ones in smaller to smaller_dict with unique IDs after larger's to count as disagreement
    for i in range(leftover_start_point + len(larger), leftover_start_point + len(larger) + len(smaller)):
        assert(i not in pair.larger_dict.keys())
        assert(i not in pair.smaller_dict.keys())
        pair.smaller_dict[i] = 4

    return (ka.krippendorff_alpha([pair.larger_dict, pair.smaller_dict], metric=ka.nominal_metric), pair)


def most_common_disagreements(larger, l_annotator, smaller, s_annotator, doc_id):
    larger_copy = deepcopy(larger)
    smaller_copy = deepcopy(smaller)
    for this_relation in larger_copy:
        # then, compare to see if unrolled versions are essentially the same
        # slightly different boundaries but the same annotation
        this_unrolled = unroll(this_relation)
        this_x_unrolled = unroll(change_x(this_relation))

        # go though all the relations in smaller and find an overlapping one
        for that_relation in smaller_copy:
            that_unrolled = unroll(that_relation)

            # check whether they overlap 
            if boundaries_overlap(this_unrolled, that_unrolled) or boundaries_overlap(unroll(flip_hard(this_relation)), that_unrolled):
                # if the labels don't match, record in disagreement dict 
                if not labels_equal(this_unrolled[2], that_unrolled[2]):
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
                    
                    '''
                    # if it's one of the top 10 disagreements, dump in disagreement_full_dict
                    if key in top_10_combinations_dict.keys():
                        top_10_combinations_dict[key] += [[doc_id, l_annotator, this_relation, s_annotator, that_relation]]
                    '''


alpha_list = []
agreement_by_pair = {
    'Sheridan and Muskaan': {'Sheridan':{}, 'Muskaan':{}}, 
    'Sheridan and Kate': {'Sheridan':{}, 'Kate':{}}, 
    'Muskaan and Kate': {'Muskaan':{}, 'Kate':{}}
}
disagreement_dict = {}
'''
# from running this code before, with ignore_elab_disagreements=False
top_10_combinations = [ 
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
'''

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

        len_a = len(a_container)
        len_b = len(b_container)
        a_container_2 = deepcopy(a_container)
        b_container_2 = deepcopy(b_container)

        if len_a >= len_b:
            score, pair = coherence_agreement(a_container, b_container)
            pair.set_which_annotator(a_annotator, b_annotator)
            most_common_disagreements(a_container_2, a_annotator, b_container_2, b_annotator, doc_id)
        else:
            score, pair = coherence_agreement(b_container, a_container)
            pair.set_which_annotator(b_annotator, a_annotator)
            most_common_disagreements(b_container_2, b_annotator, a_container_2, a_annotator, doc_id)
            
        proportion_a = pair.number_overlapping / len_a
        proportion_b = pair.number_overlapping / len_b
        alpha_list += [[doc_id, 'human' if is_human else 'grover', score, a_annotator, b_annotator, len_a, len_b, pair.number_matching, pair.number_overlapping, proportion_a, proportion_b]]

        # concatenate onto large vectors for each pair of annotators 
        if (a_annotator == 'Sheridan' and b_annotator == 'Muskaan') or (b_annotator == 'Sheridan' and a_annotator == 'Muskaan'):
            p = agreement_by_pair['Sheridan and Muskaan']
        elif (a_annotator == 'Sheridan' and b_annotator == 'Kate') or (b_annotator == 'Sheridan' and a_annotator == 'Kate'):
            p = agreement_by_pair['Sheridan and Kate']
        elif (a_annotator == 'Muskaan' and b_annotator == 'Kate') or (b_annotator == 'Muskaan' and a_annotator == 'Kate'):
            p = agreement_by_pair['Muskaan and Kate']
        
        # shift up all of the keys of the current containers so they can be appended 
        a_current_dict = pair.get_dict_by_annotator(a_annotator)
        b_current_dict = pair.get_dict_by_annotator(b_annotator)
        a_shifted = {}
        b_shifted = {}
        try:
            shift = max(max(p[a_annotator].keys()), max(p[b_annotator].keys()))
        except ValueError:
            shift = 0

        for k,v in a_current_dict.items():
            a_shifted[k + shift] = v
        for k,v in b_current_dict.items():
            b_shifted[k + shift] = v
        assert(len(a_shifted) > 0 and len(b_shifted) > 0)

        assert(len(p[a_annotator].keys() & a_shifted.keys()) == 0)
        assert(len(p[b_annotator].keys() & b_shifted.keys()) == 0)
        p[a_annotator].update(a_shifted)
        p[b_annotator].update(b_shifted)


print("boundaries_lenient=" + str(boundaries_lenient))
print("ignore_elab_disagreements=" + str(ignore_elab_disagreements))
alpha_scores = pd.DataFrame(alpha_list, columns=['doc_id', 'type', 'kripp_alpha', 'a_annotator', 'b_annotator', 'a_num_annotations', 'b_num_annotations', 'number_matching', 'number_overlapping', 'proportion_overlapping_a', 'proportion_overlapping_b'])
#print(alpha_scores.sort_values('kripp_alpha'))
print(alpha_scores[['doc_id', 'type', 'kripp_alpha', 'a_annotator', 'b_annotator', 'a_num_annotations', 'b_num_annotations', 'number_matching', 'proportion_overlapping_a', 'proportion_overlapping_b']])

print("overall mean alpha score: ", alpha_scores['kripp_alpha'].mean())
print("human mean alpha score: ", alpha_scores[(alpha_scores['type'] == 'human')]['kripp_alpha'].mean())
print("grover mean alpha score: ", alpha_scores[(alpha_scores['type'] == 'grover')]['kripp_alpha'].mean())

# concatenate docs together to calculate agreement for each pair of annotators instead of means for each pair
print('\n' + 'agreement concatenating docs together:')
for k in agreement_by_pair.keys():
    a, b = agreement_by_pair[k].keys()
    print(k, ka.krippendorff_alpha([agreement_by_pair[k][a], agreement_by_pair[k][b]], metric=ka.nominal_metric, convert_items=str))



disagreement_df = pd.DataFrame.from_dict({'combination' : disagreement_dict.keys(), 'count' : disagreement_dict.values()})
pd.set_option('display.max_colwidth', None)
print('\n', disagreement_df.sort_values('count', ascending=False).head(10))

'''
# randomly choose 10 of each type of combination from top_10_combinations_dict
for key in top_10_combinations_dict.keys():
    lst = top_10_combinations_dict[key]
    #sample = random.sample(lst, min(10, len(lst)))
    sample = lst #actually, let's just take the whole list

    df = pd.DataFrame(sample, columns=['doc_id', 'a_annotator', 'a_relation', 'b_annotator', 'b_relation'])
    #df.to_csv('100_coh_disagreements/' + key + '.csv')
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


