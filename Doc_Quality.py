# older code. Calculates the distribution of document-level ratings for human and
# Grover documents

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
        for line in annotated_doc:
            if line.strip() != "":
                if "***" in line:
                    info_container[number].append(line.strip().replace('***','').split())

G_narr_count = 0
G_narr_ratings = []
G_arg_count = 0
G_att_counts = {i:0 for i in range(6)}
H_narr_count = 0
H_narr_ratings = []
H_arg_count = 0
H_att_counts = {i:0 for i in range(6)}
for element in info_container.keys():
    if info_container[element][0][0] == 'grover':
        if info_container[element][1][0] == 'YES':
            G_narr_count += 1
            narr_ratings = [int(i) for i in info_container[element][1][1:5]]
            G_narr_ratings.append(narr_ratings)
        G_att_counts[int(info_container[element][1][9])] +=1
        if int(info_container[element][1][9]) != 0:
            G_arg_count += 1
    else:
        if info_container[element][1][0] == 'YES':
            H_narr_count += 1
            narr_ratings = [int(i) for i in info_container[element][1][1:5]]
            H_narr_ratings.append(narr_ratings)
        H_att_counts[int(info_container[element][1][9])] +=1
        if int(info_container[element][1][9]) != 0:
            H_arg_count += 1

print(G_narr_count)
print(np.mean(G_narr_ratings,axis=0))
print(H_narr_count)
print(np.mean(H_narr_ratings,axis=0))
print(G_arg_count)
print(G_att_counts)
print(H_arg_count)
print(H_att_counts)
