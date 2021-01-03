# this file contains the code that fills in all of the containers for SE types,
# coherence relations, and doc-level ratings. used in Smith_test.py, agreement.py

import csv
import os
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"
annotators = {"Sheridan":[],"Muskaan":[],"Kate":[]}

# function that fills in h_docs and g_docs
def fill_in_human_grover(h_docs, g_docs):
    with open("info.csv","r") as metadata:
        reader = csv.reader(metadata)
        for id_,line in enumerate(reader):
            if id_ != 0 and len(line) > 3:
                if line[3] == "original":
                    h_docs.append(int(line[0].strip()))
                elif line[3] == "grover":
                    g_docs.append(int(line[0].strip()))


# function that takes in the containers and fills them in using corpus
def fill_in_containers(h_docs, g_docs, G_SE_container, G_Coh_container, 
G_Doc_container, H_SE_container, H_Coh_container, H_Doc_container, SE_accounted_for,
Coh_accounted_for, doc_counter):
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
                        else:
                            try:
                                s = line.strip().split('##')[1].split('//')[0]
                                SE_temp_container.append(s.strip('\* '))
                            except:
                                pass

                # record SE ratings
                if doc_id in h_docs:
                    H_SE_container[annotator][doc_id] = SE_temp_container
                elif doc_id in g_docs:
                    G_SE_container[annotator][doc_id] = SE_temp_container
                else:
                    raise Exception("The document index could not be found.")

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

                            if doc_id in h_docs:
                                if doc_id not in H_Coh_container[annotator]:
                                    H_Coh_container[annotator][doc_id] = [line.strip().split('//')[0].split()]
                                else:
                                    H_Coh_container[annotator][doc_id].append(line.strip().split('//')[0].split())
                            elif doc_id in g_docs:
                                if doc_id not in G_Coh_container[annotator]:
                                    G_Coh_container[annotator][doc_id] = [line.strip().split('//')[0].split()]
                                else:
                                    G_Coh_container[annotator][doc_id].append(line.strip().split('//')[0].split())
                            else:
                                raise Exception("The document index could not be found.")

                    Coh_accounted_for.append(doc_id)

    return doc_counter