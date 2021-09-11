# this file contains the code that fills in all of the containers for SE types,
# coherence relations, and doc-level ratings. used in 3.Smith_test.py, agreement.py

import csv
import os
from itertools import groupby
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"
annotators = {"0":[],"2":[],"1":[]}

# function that fills in h_docs and g_docs
def fill_in_human_grover(h_docs, g_docs, d_docs):
    with open("1.info.csv", "r") as metadata:
        reader = csv.reader(metadata)
        for id_,line in enumerate(reader):
            if id_ != 0 and len(line) > 3:
                if line[3] == "original":
                    h_docs.append(int(line[0].strip()))
                elif line[3] == "grover":
                    g_docs.append(int(line[0].strip()))
                elif line[3] == "davinci":
                    d_docs.append(int(line[0].strip()))
    
    #temporarily, the two docs that I manually added, add them here
    h_docs.append(50121172355)
    g_docs.append(50121173310) #put back after davinci is done


# function that takes in the containers and fills them in using corpus
def fill_in_containers(h_docs, g_docs, d_docs, G_SE_container, G_Coh_container,
G_Doc_container, H_SE_container, H_Coh_container, H_Doc_container, D_SE_container, D_Coh_container,
D_Doc_container, SE_accounted_for, Coh_accounted_for, doc_counter, remove_low_confidence=False):
    for annotator in annotators:

        folder = os.listdir("{}/".format(annotator))

        SE_files = [file for file in folder if 'annotation' not in file]

        # gather all the SE types in each file
        for file in SE_files:

            # get the document ID
            doc_id = int(file.replace('.txt',''))

            # record the SE types annotated for that document in appropriate container
            with open(path+("{}/".format(annotator))+file,'r', encoding="utf-8") as annotated_doc:

                # initialize container for the SE types
                SE_temp_container = []
                for line in annotated_doc:

                    if line.strip() != "":

                        # save document-level ratings
                        if "***" in line:
                            if doc_id in h_docs:
                                H_Doc_container[annotator][doc_id] = line.strip().replace('***','').split()
                            elif doc_id in g_docs:
                                G_Doc_container[annotator][doc_id] = line.strip().replace('***','').split()
                            elif doc_id in d_docs:
                                D_Doc_container[annotator][doc_id] = line.strip().replace('***','').split()
                        else:
                            try:
                                label_contents = line.strip().split('##')[1].split('//')
                                if len(label_contents) > 1 and remove_low_confidence:
                                    SE_temp_container.append('*') # append 'missing_item' placeholder *
                                else:
                                    s = label_contents[0]
                                    SE_temp_container.append(s.strip('\* '))
                            except:
                                pass

                # record SE ratings
                if doc_id in h_docs:
                    H_SE_container[annotator][doc_id] = SE_temp_container
                elif doc_id in g_docs:
                    G_SE_container[annotator][doc_id] = SE_temp_container
                elif doc_id in d_docs:
                    D_SE_container[annotator][doc_id] = SE_temp_container
                else:
                    pass
                    #raise Exception("The document index could not be found.")

            #if doc_id is not already accounted for, add to counter of unique docs
            if not [t for t in SE_accounted_for if t[0]==doc_id]: 
                doc_counter += 1 
            
            # if doc_id has been seen before, need to adjudicate between them 
            # assumes that line numbers are the same 
            else:
                # TODO: adjudicate between the two versions 
                '''
                # there should only be one other doc, take the first element
                that_doc_id, that_annotator, is_human = [t for t in SE_accounted_for if t[0]==doc_id][0]

                # retrieve containers for SE types for both versions of doc
                if is_human:
                    this_container = H_SE_container[annotator][doc_id] 
                    that_container = H_SE_container[that_annotator][that_doc_id]
                else:
                    this_container = G_SE_container[annotator][doc_id] 
                    that_container = G_SE_container[that_annotator][that_doc_id]

                assert(len(this_container) == len(that_container))
                '''

            #add document to SE_accounted_for, with annotator and whether it's human
            SE_accounted_for.append((doc_id, annotator, doc_id in h_docs))

                
        # list coherence ratings
        Coh_files = [file for file in folder if 'annotation' in file]

        # for each coherence relations file
        for file in Coh_files:

            # find the id
            doc_id = int(file.replace('.txt-annotation',''))

            if doc_id not in Coh_accounted_for:
                with open(path+("{}/".format(annotator))+file,'r') as annotated_doc:

                    # record the coherence relations
                    for line in annotated_doc:
                        if line.strip() != "":
                            relation = line.strip().split('//')[0].split()
                            if not (len(line.strip().split('//')) > 1 and remove_low_confidence): 
                                if doc_id in h_docs:
                                    if doc_id not in H_Coh_container[annotator]:
                                        H_Coh_container[annotator][doc_id] = [relation]
                                    else:
                                        H_Coh_container[annotator][doc_id].append(relation)
                                elif doc_id in g_docs:
                                    if doc_id not in G_Coh_container[annotator]:
                                        G_Coh_container[annotator][doc_id] = [relation]
                                    else:
                                        G_Coh_container[annotator][doc_id].append(relation)
                                elif doc_id in d_docs:
                                    if doc_id not in D_Coh_container[annotator]:
                                        D_Coh_container[annotator][doc_id] = [relation]
                                    else:
                                        D_Coh_container[annotator][doc_id].append(relation)
                                else:
                                    pass
                                    #raise Exception("The document index could not be found.")

                    Coh_accounted_for.append((doc_id, annotator, doc_id in h_docs))

    # get rid of duplicate annotations for coherence relations
    for annotator in annotators:
        remove_duplicates = lambda l: list(k for k,_ in groupby(l))
        G_Coh_container[annotator] = {k: remove_duplicates(v) for k,v in G_Coh_container[annotator].items()}
        H_Coh_container[annotator] = {k: remove_duplicates(v) for k,v in H_Coh_container[annotator].items()}

    return doc_counter



# function that fills in more robust SE containers, keeping the text with the annotations
def fill_in_SE_robust(h_docs, g_docs, d_docs, G_SE_container, H_SE_container, D_SE_container, SE_accounted_for, doc_counter):
    for annotator in annotators:

        folder = os.listdir("{}/".format(annotator))

        SE_files = [file for file in folder if 'annotation' not in file]

        # gather all the SE types in each file
        for file in SE_files:

            # get the document ID
            doc_id = int(file.replace('.txt',''))

            # record the SE types annotated for that document in appropriate container
            with open(path+("{}/".format(annotator))+file,'r', encoding="utf-8") as annotated_doc:

                # initialize container for the SE types
                SE_temp_container = []
                for line in annotated_doc:

                    if line.strip() != "":
                        if "***" not in line:
                            try:
                                text = line.strip().split('##')[0]
                                s = line.strip().split('##')[1].split('//')[0]
                                SE_temp_container += [(s.strip('\* '), text)]
                            except:
                                pass

                # record SE ratings
                if doc_id in h_docs:
                    H_SE_container[annotator][doc_id] = SE_temp_container
                elif doc_id in g_docs:
                    G_SE_container[annotator][doc_id] = SE_temp_container
                elif doc_id in d_docs:
                    D_SE_container[annotator][doc_id] = SE_temp_container
                else:
                    print(doc_id)
                    #raise Exception("The document index could not be found.")


            #if doc_id is not already accounted for, add to counter of unique docs
            if not [t for t in SE_accounted_for if t[0]==doc_id]: 
                doc_counter += 1 
            
            # if doc_id has been seen before, need to adjudicate between them 
            # assumes that line numbers are the same 
            else:
                # TODO: adjudicate between the two versions 
                '''
                # there should only be one other doc, take the first element
                that_doc_id, that_annotator, is_human = [t for t in SE_accounted_for if t[0]==doc_id][0]

                # retrieve containers for SE types for both versions of doc
                if is_human:
                    this_container = H_SE_container[annotator][doc_id] 
                    that_container = H_SE_container[that_annotator][that_doc_id]
                else:
                    this_container = G_SE_container[annotator][doc_id] 
                    that_container = G_SE_container[that_annotator][that_doc_id]

                assert(len(this_container) == len(that_container))
                '''

            #add document to SE_accounted_for, with annotator and whether it's human
            SE_accounted_for.append((doc_id, annotator, doc_id in h_docs))

           
    return doc_counter
