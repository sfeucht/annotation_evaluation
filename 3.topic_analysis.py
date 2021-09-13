# produces correlations for frequency of topic words vs narrative presence (line #), topic vs argument presence (#)
# topic vs narrative quality measures (#), and topic vs argument quality measures (#)

import csv
import re
import os
import sys

import pandas as pd
import scipy
from scipy.stats import pearsonr
from csv import DictReader
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
G_Doc_container, H_SE_container, H_Coh_container, H_Doc_container, D_SE_container, D_Coh_container, D_Doc_container,
SE_accounted_for, Coh_accounted_for, doc_counter)

news_column_headers = ['doc_id', 'Internet', 'US administration', 'Legal procedures', 'Military', 'Locations',
                       'Quantitative', 'Images', 'Time', 'Law enforcement', 'Educational institutions',
                       'Body, eating and weight', 'Justice system', 'Countries', 'UNCLEAR (British focus)',
                       'Familial relationships', 'Ubiquitous terms (go, get, think, etc.)', 'Organized crime',
                       'Scientific research (?)', 'Sports', 'Medical', 'States', 'Army procedures', 'UNCLEAR (Wedding/terrorism)',
                       'Attacks', 'Natural environment', 'Family', 'Violence and race', 'Public statements',
                       'Reports and their dates', 'UNCLEAR (animals/Hollywood)', 'Hospital and procedures', 'Finance',
                       'Churches, regions', 'Manufacturing (?)', 'Energy and mining', 'Technology', 'Market forces',
                       'Research', 'Immigration and minorities', 'Job market', 'Marijuana', 'Cars and crashes',
                       'UNCLEAR', 'Food', 'Fire', 'Party procedures', 'Party politics', 'Social media content',
                       'TV and movies', 'Weather']

news_topics = ['Internet', 'US administration', 'Legal procedures', 'Military', 'Locations',
                       'Quantitative', 'Images', 'Time', 'Law enforcement', 'Educational institutions',
                       'Body, eating and weight', 'Justice system', 'Countries', 'UNCLEAR (British focus)',
                       'Familial relationships', 'Ubiquitous terms (go, get, think, etc.)', 'Organized crime',
                       'Scientific research (?)', 'Sports', 'Medical', 'States', 'Army procedures', 'UNCLEAR (Wedding/terrorism)',
                       'Attacks', 'Natural environment', 'Family', 'Violence and race', 'Public statements',
                       'Reports and their dates', 'UNCLEAR (animals/Hollywood)', 'Hospital and procedures', 'Finance',
                       'Churches, regions', 'Manufacturing (?)', 'Energy and mining', 'Technology', 'Market forces',
                       'Research', 'Immigration and minorities', 'Job market', 'Marijuana', 'Cars and crashes',
                       'UNCLEAR', 'Food', 'Fire', 'Party procedures', 'Party politics', 'Social media content',
                       'TV and movies', 'Weather']

reddit_column_headers = ['doc_id', 'Effects/prohibition vis-a-vis alcohol/tobacco', 'UNCLEAR1', 'Hemp: legality and uses',
                         'Party politics and ideology', 'Quantities and limits', 'Legal status and common use',
                         'Govt. power and ind. rights/freedoms', 'Border, organized crime & Latin influence',
                         'State vs. federal legalization', 'Media campaigns & portrayals', 'enforcement vis-a-vis violent crimes',
                         'Legal market & economic forces', 'Addiction potential & gateway status', 'Reasoning and arguments',
                         'State-level legal. timelines', 'Police car searches', 'Medical marijuana effects and access',
                         'Cannabis types and use methods', 'Marijuana use and the workplace', 'FDA schedules', 'UNCLEAR2',
                         'Continuity of US drug & foreign policy', 'Age and familial relations', 'Marijuana and finances',
                         'User stereotypes and life outcomes', 'Private interests & the prison industry', 'UNCLEAR3',
                         'Legalization across US and the world', 'Police house searches & seizure', 'Legal procedures',
                         'Emotional and life impact', 'Reddit moderation', 'Everyday enforcement encounters', 'UNCLEAR4', 'UNCLEAR5',
                         'Drug testing', 'Judgments of character', 'Imprisonment over marijuana', 'Electoral politics & parties',
                         'UNCLEAR6', 'Local/state regulations', 'Health and opinion research', 'DUI effects & enforcement',
                         'Racial/minority disparities', 'Federal Court Processes', 'Smoking methods, health effects and bans',
                         'UNCLEAR7', 'Enforcement & observance', 'Gun versus marijuana regulations', 'Expletives-laden discourse']

reddit_topics = ['Effects/prohibition vis-a-vis alcohol/tobacco', 'UNCLEAR1', 'Hemp: legality and uses',
                         'Party politics and ideology', 'Quantities and limits', 'Legal status and common use',
                         'Govt. power and ind. rights/freedoms', 'Border, organized crime & Latin influence',
                         'State vs. federal legalization', 'Media campaigns & portrayals', 'enforcement vis-a-vis violent crimes',
                         'Legal market & economic forces', 'Addiction potential & gateway status', 'Reasoning and arguments',
                         'State-level legal. timelines', 'Police car searches', 'Medical marijuana effects and access',
                         'Cannabis types and use methods', 'Marijuana use and the workplace', 'FDA schedules', 'UNCLEAR2',
                         'Continuity of US drug & foreign policy', 'Age and familial relations', 'Marijuana and finances',
                         'User stereotypes and life outcomes', 'Private interests & the prison industry', 'UNCLEAR3',
                         'Legalization across US and the world', 'Police house searches & seizure', 'Legal procedures',
                         'Emotional and life impact', 'Reddit moderation', 'Everyday enforcement encounters', 'UNCLEAR4', 'UNCLEAR5',
                         'Drug testing', 'Judgments of character', 'Imprisonment over marijuana', 'Electoral politics & parties',
                         'UNCLEAR6', 'Local/state regulations', 'Health and opinion research', 'DUI effects & enforcement',
                         'Racial/minority disparities', 'Federal Court Processes', 'Smoking methods, health effects and bans',
                         'UNCLEAR7', 'Enforcement & observance', 'Gun versus marijuana regulations', 'Expletives-laden discourse']

def annotator_tag(Doc_container): # tags every doc_id with its annotator number

    tagged_dict = {}

    for annotator in Doc_container.keys():

        tagged_dict[annotator] = {}

        for k,v in Doc_container[annotator].items():

            tagged_dict[annotator][str(k) + str(annotator)] = v

    return tagged_dict

# applying the tags to the containers

#H_Doc_container = annotator_tag(H_Doc_container)
#G_Doc_container = annotator_tag(G_Doc_container)
#D_Doc_container = annotator_tag(D_Doc_container)
#H_Coh_container = annotator_tag(H_Coh_container)
#G_Coh_container = annotator_tag(G_Coh_container)
#D_Coh_container = annotator_tag(D_Coh_container)
#H_SE_container = annotator_tag(H_SE_container)
#G_SE_container = annotator_tag(G_SE_container)
#D_SE_container = annotator_tag(D_SE_container)

news_not_normalized = pd.read_csv(r'/Users/ravram1836gmail.com/Desktop/annotation_evaluation.nosync/3.news_wc_not_normalized.csv')
news_not_normalized.columns = news_column_headers
reddit_not_normalized = pd.read_csv(r'/Users/ravram1836gmail.com/Desktop/annotation_evaluation.nosync/3.reddit_wc_not_normalized.csv')
reddit_not_normalized.columns = reddit_column_headers

_150_news_not_normalized = pd.read_csv(r'/Users/ravram1836gmail.com/Desktop/annotation_evaluation.nosync/3.news_Ali_150_wc_not_normalized.csv')
_150_topics_csv = pd.read_csv(r'/Users/ravram1836gmail.com/Desktop/annotation_evaluation.nosync/3.150 topic model - Sheet1.csv')
_150_topics = _150_topics_csv['title'].to_list()
_150_column_headers = ['doc_id'] + _150_topics
_150_news_not_normalized.columns = _150_column_headers

_250_news_not_normalized = pd.read_csv(r'/Users/ravram1836gmail.com/Desktop/annotation_evaluation.nosync/3.news_Ali_250_wc_not_normalized.csv')
_250_topics_csv = pd.read_csv(r'/Users/ravram1836gmail.com/Desktop/annotation_evaluation.nosync/3.250 topic model - Sheet1.csv')
_250_topics = _250_topics_csv['title'].to_list()
_250_column_headers = ['doc_id'] + _250_topics
_250_news_not_normalized.columns = _250_column_headers

info = pd.read_csv(r'/Users/ravram1836gmail.com/Desktop/annotation_evaluation.nosync/1.info.csv')

def generate_domain_dict(info_file): # produces a dict that records doc IDs and domains from given info file, used later
# in code to identify documents as reddit or news

    domains = [x for x in info_file['domain'].to_list() if str(x) != 'nan'] # creates list of document sources
    ids = [x for x in info_file['id'].to_list() if str(x) != 'COLLECTED'] # creates a list of document IDs

    domain_dict = {}

    for id in ids: # creates dict with ID as key, domain as value
        for domain in domains:
            domain_dict[int(id)] = domain
            domains.remove(domain)
            break

    del domain_dict[50121172355] # FOR BABAK: this is old code that dealt with duplicates... can delete now that we have annotator tags?
    domain_dict[501211723551] = 'washingtonpost.com'
    domain_dict[501211723550] = 'washingtonpost.com'

    del domain_dict[241120231449]
    domain_dict[2411202314491] = 'foxnews.com'
    domain_dict[2411202314490] = 'foxnews.com'

    del domain_dict[291220224329]
    domain_dict[2912202243291] = 'reddit.com/r/news'
    domain_dict[2912202243290] = 'reddit.com/r/news'

    return domain_dict

domain_dictionary = generate_domain_dict(info)

########################################################
### TOPIC CORRELATION WITH PRESENCE OF NARRATIVE

def p_of_narrative_topic_nums(topic_csv_file, Doc_container, topic): # uses a topic csv file to produce a dict with
# counts of a given topic for each document, called by p_of_nar_topic_correl_calc

    topic_dict = {}

    id_list = topic_csv_file['doc_id'].to_list() # creates a list of doc IDs from the topic csv file
    topic_nums = topic_csv_file[topic].to_list() # creates a list of topic counts

    for id in id_list:

        id = int(id.translate({ord(i): None for i in '.txt'}))

        for num in topic_nums: # creates a dict with a doc ID as the key, topic count as the value
            topic_dict[id] = num
            topic_nums.remove(num)
            break

        if id not in Doc_container['1'].keys() and id not in Doc_container['2'].keys() \
                and id not in Doc_container['0'].keys():
                    del topic_dict[id] #deletes documents that aren't in the Doc_container to avoid index errors

        for d_id in domain_dictionary.keys(): #deletes all reddit or all news, depending on which analysis is being done

            if d_id in topic_dict.keys():

                if 'Weather' in topic_csv_file.columns and 'reddit' in domain_dictionary[d_id]:
                    del topic_dict[d_id] #uses Weather because this is unique to the news topics

                if 'Reddit moderation' in topic_csv_file.columns and 'reddit' not in domain_dictionary[d_id]:
                    del topic_dict[d_id] #uses Reddit moderation because this is unique to the reddit topics

    return topic_dict

def presence_of_narrative(Doc_container, t_dict, index): # produces a dict of narrative presence for each document (with 0 for
# no narrative and 1 for some narrative), called by p_of_nar_topic_correl_calc

    rating_dict = {}

    for topic_id in t_dict.keys():
        for annotator in Doc_container.keys():
            for doc_id in Doc_container[annotator].keys():

                ratings = Doc_container[annotator][doc_id]

                if doc_id == topic_id: # to match IDs across dicts

                    if (ratings[index] == 'NA') or (ratings[index] == '0'): # adds 0 for no narrative
                        if(doc_id not in rating_dict):
                            rating_dict[doc_id] = {}
                            rating_dict[doc_id] = 0
                    if (ratings[index] != 'NA') and (ratings[index] != '0'): # adds 1 for some narrative
                        if(doc_id not in rating_dict):
                            rating_dict[doc_id] = {}
                            rating_dict[doc_id] = 1
    return rating_dict

def p_of_nar_topic_correl_calc(topic_csv_file, topics, Doc_container): # uses both of the previous functions to find
# correlations between each topic and presence of narrative

    for t in topics:
        topic_dict = p_of_narrative_topic_nums(topic_csv_file, Doc_container, t)
        rating_dict = presence_of_narrative(Doc_container, topic_dict, 3)
        topic_list = []
        rating_list = []

        for k,v in rating_dict.items(): # creates a list of narrative presence for pointbiserialr
            rating_list.append(int(v))

        for k1,v1 in topic_dict.items(): # creates a list of topic counts for pointbiserialr
            topic_list.append(v1)

        r,p = scipy.stats.pointbiserialr(rating_list, topic_list)

        if abs(r) > 0.2:
            print(t, 'r =', round(r, 2), 'p =', round(p, 2), '{', np.count_nonzero(topic_list), 'non-zero values }')

    return

# 150 news topics
#print(p_of_nar_topic_correl_calc(_150_news_not_normalized, _150_topics, H_Doc_container))
#print(p_of_nar_topic_correl_calc(_150_news_not_normalized, _150_topics, G_Doc_container))

# 250 news topics
#print(p_of_nar_topic_correl_calc(_250_news_not_normalized, _250_topics, H_Doc_container))
#print(p_of_nar_topic_correl_calc(_250_news_not_normalized, _250_topics, G_Doc_container))

# 50 news/ reddit topics, grover and human
#print(p_of_nar_topic_correl_calc(news_not_normalized, news_topics, H_Doc_container))
#print(p_of_nar_topic_correl_calc(reddit_not_normalized, reddit_topics, H_Doc_container))
#print(p_of_nar_topic_correl_calc(news_not_normalized, news_topics, G_Doc_container))
#print(p_of_nar_topic_correl_calc(reddit_not_normalized, reddit_topics, G_Doc_container))

###########################################################

########################################################
### TOPIC CORRELATION WITH PRESENCE OF ARGUMENT

def p_of_argument_topic_nums(topic_csv_file, Doc_container, topic):  # uses a topic csv file to produce a dict with
# counts of a given topic for each document, called by p_of_arg_topic_correl_calc

    topic_dict = {}

    id_list = topic_csv_file['doc_id'].to_list() # creates a list of doc IDs from the topic csv file
    topic_nums = topic_csv_file[topic].to_list() # creates a list of topic counts

    for id in id_list:

        id = int(id.translate({ord(i): None for i in '.txt'}))

        for num in topic_nums: # creates a dict with a doc ID as the key, topic count as the value
            topic_dict[id] = num
            topic_nums.remove(num)
            break

        if id not in Doc_container['1'].keys() and id not in Doc_container['3'].keys() \
                and id not in Doc_container['0'].keys():
                    del topic_dict[id] #deletes documents that aren't in the Doc_container to avoid index errors

        for d_id in domain_dictionary.keys(): #deletes all reddit or all news, depending on which analysis is being done

            if d_id in topic_dict.keys():

                if 'Weather' in topic_csv_file.columns and 'reddit' in domain_dictionary[d_id]:
                    del topic_dict[d_id] #uses Weather because this is unique to the news topics

                if 'Reddit moderation' in topic_csv_file.columns and 'reddit' not in domain_dictionary[d_id]:
                    del topic_dict[d_id] #uses Reddit moderation because this is unique to the reddit topics

    return topic_dict

def presence_of_argument(Doc_container, t_dict, index): # produces a dict of narrative presence for each document (with 0 for
# no narrative and 1 for some narrative), called by p_of_arg_topic_correl_calc

    rating_dict = {}

    for topic_id in t_dict.keys():
        for annotator in Doc_container.keys():
            for doc_id in Doc_container[annotator].keys():

                ratings = Doc_container[annotator][doc_id]

                if doc_id == topic_id: # to match IDs across dicts

                    if (ratings[index] == 'NA') or (ratings[index] == '0'): # adds 0 for no argument
                        if(doc_id not in rating_dict):
                            rating_dict[doc_id] = {}
                            rating_dict[doc_id] = 0
                    if (ratings[index] != 'NA') and (ratings[index] != '0'): # adds 1 for some argument
                        if(doc_id not in rating_dict):
                            rating_dict[doc_id] = {}
                            rating_dict[doc_id] = 1
    return rating_dict

def p_of_arg_topic_correl_calc(topic_csv_file, topics, Doc_container): # uses both of the previous functions to find
# correlations between each topic and presence of argument

    for t in topics:
        topic_dict = p_of_argument_topic_nums(topic_csv_file, Doc_container, t)
        rating_dict = presence_of_argument(Doc_container, topic_dict, 12)
        topic_list = []
        rating_list = []

        for k,v in rating_dict.items(): # creates a list of argument presence for pointbiserialr
            rating_list.append(int(v))

        for k1,v1 in topic_dict.items(): # creates a list of topic counts for pointbiserialr
            topic_list.append(v1)

        r,p = scipy.stats.pointbiserialr(rating_list, topic_list)

        if abs(r) >= 0.2:
            print(t, 'r =', round(r, 2), 'p =', round(p, 2), '{', np.count_nonzero(topic_list), 'non-zero values }')

    return

# 150 news topics
#print(p_of_arg_topic_correl_calc(_150_news_not_normalized, _150_topics, H_Doc_container))
#print(p_of_arg_topic_correl_calc(_150_news_not_normalized, _150_topics, G_Doc_container))

# 250 news topics
#print(p_of_arg_topic_correl_calc(_250_news_not_normalized, _250_topics, H_Doc_container))
#print(p_of_arg_topic_correl_calc(_250_news_not_normalized, _250_topics, G_Doc_container))

# 50 news/ reddit topics, human and grover
#print(p_of_arg_topic_correl_calc(news_not_normalized, news_topics, H_Doc_container))
#print(p_of_arg_topic_correl_calc(reddit_not_normalized, reddit_topics, H_Doc_container))
#print(p_of_arg_topic_correl_calc(news_not_normalized, news_topics, G_Doc_container))
#print(p_of_arg_topic_correl_calc(reddit_not_normalized, reddit_topics, G_Doc_container))

########################################################

########################################################
### TOPIC CORRELATION WITH NARRATIVE QUALITY MEASURES

def narrative_topic_nums(topic_csv_file, Doc_container, topic, index): # uses a topic csv file to produce a dict with
# counts of a given topic for each document with some narrative, called by nar_topic_correl_calc

    topic_dict = {}

    id_list = topic_csv_file['doc_id'].to_list() # creates a list of doc IDs from the topic csv file
    topic_nums = topic_csv_file[topic].to_list() # creates a list of topic counts

    for id in id_list:

        id = int(id.translate({ord(i): None for i in '.txt'}))

        for num in topic_nums: # creates a dict with a doc ID as the key, topic count as the value
            topic_dict[id] = num
            topic_nums.remove(num)
            break

        if id not in Doc_container['1'].keys() and id not in Doc_container['2'].keys() \
                and id not in Doc_container['0'].keys():
                    del topic_dict[id] #deletes documents that aren't in the Doc_container to avoid index errors

        for d_id in domain_dictionary.keys(): #deletes all reddit or all news, depending on which analysis is being done

            if d_id in topic_dict.keys():

                if 'Weather' in topic_csv_file.columns and 'reddit' in domain_dictionary[d_id]:
                    del topic_dict[d_id] #uses Weather because this is unique to the news topics

                if 'Reddit moderation' in topic_csv_file.columns and 'reddit' not in domain_dictionary[d_id]:
                    del topic_dict[d_id] #uses Reddit moderation because this is unique to the reddit topics

    for annotator in Doc_container.keys():
        for doc_id in Doc_container[annotator].keys(): # dropping docs without narrative

            ratings = Doc_container[annotator][doc_id]

            if doc_id in topic_dict.keys():

                if (ratings[index] == 'NA') or (ratings[index] == '0'):
                    del topic_dict[doc_id]

    return topic_dict

def narrative_ratings(Doc_container, t_dict, index): # produces a dict of narrative quality measures for each document,
# called by nar_topic_correl_calc

    rating_dict = {}

    for topic_id in t_dict.keys():
        for annotator in Doc_container.keys():
            for doc_id in Doc_container[annotator].keys():

                ratings = Doc_container[annotator][doc_id]

                if doc_id == topic_id: # to match IDs across dicts

                    if (ratings[index] != 'NA') and (ratings[index] != '0'): # creating a dict doc ID as key, narrative quality measure as value
                        if(doc_id not in rating_dict):
                            rating_dict[doc_id] = {}
                        if(ratings[index] not in rating_dict[doc_id]):
                            rating_dict[doc_id] = ratings[index]
    return rating_dict

def nar_topic_correl_calc(topic_csv_file, topics, Doc_container): # uses both of the previous functions to find
    # correlations between each topic and narrative quality measures

    indexes = [4,5,6,7]

    for index in indexes:
        for t in topics:
            topic_dict = narrative_topic_nums(topic_csv_file, Doc_container, t, index)
            rating_dict = narrative_ratings(Doc_container, topic_dict, index)
            topic_list = []
            rating_list = []

            for k,v in rating_dict.items(): # creating a list of quality measures for pearsonr
                rating_list.append(int(v))

            for k1,v1 in topic_dict.items(): # creating a list of topic counts for pearsonr
                topic_list.append(v1)

            r,p = scipy.stats.pearsonr(topic_list, rating_list)

            if index == 4 and abs(r) >= 0.2:
                print(t, 'r = ', round(r, 2), 'p = ', round(p, 2), '{', np.count_nonzero(topic_list), 'non-zero values }')

            if index == 5 and abs(r) >= 0.2:
                print(t, 'r = ', round(r, 2), 'p = ', round(p, 2), '{', np.count_nonzero(topic_list), 'non-zero values }')

            if index == 6 and abs(r) >= 0.2:
                print(t, 'r = ', round(r, 2), 'p = ', round(p, 2), '{', np.count_nonzero(topic_list), 'non-zero values }')

            if index == 7 and abs(r) >= 0.2:
                print(t, 'r = ', round(r, 2), 'p = ', round(p, 2), '{', np.count_nonzero(topic_list), 'non-zero values }')

    return

# 50 news/ reddit topics, human and grover
#print(nar_topic_correl_calc(news_not_normalized, news_topics, H_Doc_container))
#print(nar_topic_correl_calc(reddit_not_normalized, reddit_topics, H_Doc_container))
#print(nar_topic_correl_calc(news_not_normalized, news_topics, G_Doc_container))
#print(nar_topic_correl_calc(reddit_not_normalized, reddit_topics, G_Doc_container))

###########################################################

########################################################
### TOPIC CORRELATION WITH ARGUMENT QUALITY MEASURES

def argument_topic_nums(topic_csv_file, Doc_container, topic, index): # uses a topic csv file to produce a dict with
# counts of a given topic for each document with some argument, called by arg_topic_correl_calc

    topic_dict = {}

    id_list = topic_csv_file['doc_id'].to_list() # creates a list of doc IDs from the topic csv file
    topic_nums = topic_csv_file[topic].to_list() # creates a list of topic counts

    for id in id_list:

        id = int(id.translate({ord(i): None for i in '.txt'}))

        for num in topic_nums: # creates a dict with a doc ID as the key, topic count as the value
            topic_dict[id] = num
            topic_nums.remove(num)
            break

        if id not in Doc_container['1'].keys() and id not in Doc_container['2'].keys() \
                and id not in Doc_container['0'].keys():
                    del topic_dict[id] #deletes documents that aren't in the Doc_container to avoid index errors

        for d_id in domain_dictionary.keys(): #deletes all reddit or all news, depending on which analysis is being done

            if d_id in topic_dict.keys():

                if 'Weather' in topic_csv_file.columns and 'reddit' in domain_dictionary[d_id]:
                    del topic_dict[d_id] #uses Weather because this is unique to the news topics

                if 'Reddit moderation' in topic_csv_file.columns and 'reddit' not in domain_dictionary[d_id]:
                    del topic_dict[d_id] #uses Reddit moderation because this is unique to the reddit topics

    for annotator in Doc_container.keys(): # dropping all docs without argument
        for doc_id in Doc_container[annotator].keys():

            ratings = Doc_container[annotator][doc_id]

            if doc_id in topic_dict.keys():

                if (ratings[index] == 'NA') or (ratings[index] == '0'):
                    del topic_dict[doc_id]

    return topic_dict

def argument_ratings(Doc_container, t_dict, index): # produces a dict of argument quality measures for each document,
# called by arg_topic_correl_calc

    rating_dict = {}

    for topic_id in t_dict.keys():
        for annotator in Doc_container.keys():
            for doc_id in Doc_container[annotator].keys():

                ratings = Doc_container[annotator][doc_id]

                if doc_id == topic_id: # to match IDs across dicts

                    if (ratings[index] != 'NA') and (ratings[index] != '0'):  # creating a dict doc ID as key, argument quality measure as value
                        if(doc_id not in rating_dict):
                            rating_dict[doc_id] = {}
                        if(ratings[index] not in rating_dict[doc_id]):
                            rating_dict[doc_id] = ratings[index]
    return rating_dict

def arg_topic_correl_calc(topic_csv_file, topics, Doc_container): # uses both of the previous functions to find
# correlations between each topic and argument quality measures

    indexes = [13,14]

    for index in indexes:
        for t in topics:
            topic_dict = argument_topic_nums(topic_csv_file, Doc_container, t, index)
            rating_dict = argument_ratings(Doc_container, topic_dict, index)
            topic_list = []
            rating_list = []

            for k,v in rating_dict.items(): # creating a list of argument quality measures for pearsonr
                rating_list.append(int(v))

            for k1,v1 in topic_dict.items(): # creating a list of topic counts for pearsonr
                topic_list.append(v1)

            r,p = pearsonr(topic_list, rating_list)

            if index == 13 and abs(r) >= 0.2:
                print(t, 'r = ', round(r, 2), 'p = ', round(p, 2), '{', np.count_nonzero(topic_list), 'non-zero values }')

            if index == 14 and abs(r) >= 0.2:
                print(t, 'r = ', round(r, 2), 'p = ', round(p, 2), '{', np.count_nonzero(topic_list), 'non-zero values }')

    return

# 50 news/ reddit topics, human and grover
#print(arg_topic_correl_calc(news_not_normalized, news_topics, H_Doc_container))
#print(arg_topic_correl_calc(reddit_not_normalized, reddit_topics, H_Doc_container))
#print(arg_topic_correl_calc(news_not_normalized, news_topics, G_Doc_container))
#print(arg_topic_correl_calc(reddit_not_normalized, reddit_topics, G_Doc_container))

###########################################################
