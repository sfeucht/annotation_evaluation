# older code for comparing the count of situation entity and coherence relation
# tags across Grover and human documents. Things like maximum lengths of relations
# are also calculated and presented on plots

import os
import re
import matplotlib.pyplot as plt
import numpy as np

regex = re.compile('[^a-zA-Z]')

file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"
SE_filelist = [file for file in os.listdir(path) if '.py' not in file and 'annotation' not in file and '.svg' not in file]
Coh_filelist = [file+"-annotation" for file in SE_filelist]

info_container = {}

for number,file in enumerate(SE_filelist):
    doc_quality = [var for var in file.replace('.txt','').split('_')]
    info_container[number] = []
    info_container[number].append(doc_quality)

    with open(path+file,'r') as annotated_doc:
        SE_container = []
        for line in annotated_doc:
            if line.strip() != "":
                if "***" not in line:
                    try:
                        SE_container.append(line.strip().split('##')[1].split('//')[0])
                    except:
                        pass
                else:
                    info_container[number].append(line.strip().replace('***','').split())
        info_container[number].append(SE_container)

for number,file in enumerate(Coh_filelist):
    coh_relations = []
    with open(path+file,'r') as annotated_doc:
        for line in annotated_doc:
            if line.strip() != "":
                coh_relations.append(regex.sub('',line))
        info_container[number].append(coh_relations)

G_SE_counts = {}
for number,doc in info_container.items():
    if doc[0][0] == 'grover':
        for EDU in doc[2]:
            if EDU.strip() in G_SE_counts.keys():
                G_SE_counts[EDU.strip()] += 1
            else:
                G_SE_counts[EDU.strip()] = 1

H_SE_counts = {}
for number,doc in info_container.items():
    if doc[0][0] != 'grover':
        for EDU in doc[2]:
            if EDU.strip() in H_SE_counts.keys():
                H_SE_counts[EDU.strip()] += 1
            else:
                H_SE_counts[EDU.strip()] = 1

print(G_SE_counts)
print(H_SE_counts)

max_num = np.max([max(list(G_SE_counts.values())),max(list(H_SE_counts.values()))])
plt.bar(G_SE_counts.keys(), G_SE_counts.values(), color='g')
plt.xticks(list(G_SE_counts.keys()),list(G_SE_counts.keys()), rotation='vertical')
plt.axis(ymax=max_num+5)
plt.xlabel("Situation Entity Type")
plt.ylabel("Frequency")
plt.title('Grover Situation Entity Frequencies')
plt.show()
plt.bar(H_SE_counts.keys(), H_SE_counts.values(), color='b')
plt.xticks(list(H_SE_counts.keys()),list(H_SE_counts.keys()), rotation='vertical')
plt.axis(ymax=max_num+5)
plt.xlabel("Situation Entity Type")
plt.ylabel("Frequency")
plt.title('Human Situation Entity Frequencies')
plt.show()

G_Coh_counts = {}
for number,doc in info_container.items():
    if doc[0][0] == 'grover':
        for EDU in doc[3]:
            if EDU.strip() in G_Coh_counts.keys():
                G_Coh_counts[EDU.strip()] += 1
            else:
                G_Coh_counts[EDU.strip()] = 1

H_Coh_counts = {}
for number,doc in info_container.items():
    if doc[0][0] != 'grover':
        for EDU in doc[3]:
            if EDU.strip() in H_Coh_counts.keys():
                H_Coh_counts[EDU.strip()] += 1
            else:
                H_Coh_counts[EDU.strip()] = 1

full_names = {'elab':'Elaboration','ce':'Cause/effect','contr':'contrast','sim':'similarity/parallel','attr':'attribution','ve':'Violated Expectation','temp':'Temporal Sequence','cond':'Condition','examp':'example','seemsdisconnected':'disconnected','doesntseemtosupport':'unsupported claim','nonelab':'Nonsensical elaboration','andunconnectedbecausesectionwascutoutfortime':'cut short'}
for short_name in G_Coh_counts.keys():
    if short_name in full_names.keys():
        G_Coh_counts[full_names[short_name]] = G_Coh_counts.pop(short_name)
for short_name in H_Coh_counts.keys():
    if short_name in full_names.keys():
        H_Coh_counts[full_names[short_name]] = H_Coh_counts.pop(short_name)

H_Coh_counts['attribution'] = H_Coh_counts.pop('attr')

print(G_Coh_counts)
print(H_Coh_counts)

max_num = np.max([max(list(G_Coh_counts.values())),max(list(H_Coh_counts.values()))])
plt.bar(G_Coh_counts.keys(), G_Coh_counts.values(), color='g')
plt.xticks(list(G_Coh_counts.keys()),list(G_Coh_counts.keys()), rotation='vertical')
# ['elaboration','attribution','same','similarity/parallel','cause/effect','contrast','example','violated expectation','condition','disconnected','unsupported claim','nonsensical elaboration']
plt.axis(ymax=max_num+5)
plt.xlabel("Coherence Relation Type")
plt.ylabel("Frequency")
plt.title('Grover Coherence Relation Frequencies')
plt.show()
plt.bar(H_Coh_counts.keys(), H_Coh_counts.values(), color='b')
plt.xticks(list(H_Coh_counts.keys()),list(H_Coh_counts.keys()), rotation='vertical')
# ['elaboration','attribution','same','similarity/parallel','cause/effect','violated expectation','temporal sequence','contrast','condition']
plt.axis(ymax=max_num+5)
plt.xlabel("Coherence Relation Type")
plt.ylabel("Frequency")
plt.title('Human Coherence Relation Frequencies')
plt.show()

print(len(SE_filelist))
print(len(Coh_filelist))
