Accompanying resources for  
**Babak Hemmatian, Sheridan Feucht, Rachel Avram, Alexander Wey, Muskaan Garg, Kate Spitalnic
Carsten Eickhoff, Ellie Pavlick, Bjorn Sandstede, Steven Sloman. 2021. A Novel Corpus of Discourse Structure in Humans and Computers. In the 2nd Workshop on Computational Approaches to Discourse at EMNLP 2021.**
See extended_abstract.pdf for the conference submission (arxiv link forthcoming)

**Corpus access**

Visit this link to access the complete corpus of annotated documents: https://drive.google.com/drive/u/1/folders/13gOFgg6cp8tQSXezj-cwML6iNn3SzU99

In this corpus, folders 0, 1, and 2 contain the documents annotated by each annotator, named here as annotator 0, 1, or 2. Files ending in .txt contain annotations for situation entities and document level ratings, while files ending in .txt-annotation contain annotations for coherence relations. More information about the annotation process can be found in the documents titled Annotation Manual and Additional Annotation Instructions.

**Guide to the files and folders on GitHub:**

Files beginning with 1. provide important background for the project.

I) annotation_key provides an example document-level annotation and breaks down what each number means.
II) annotator_profiles provides demographic information about the annotators that worked on this project.
III) document_metadata.csv lists every annotated document's ID, the date it was annotated, what type of document it is (human,
grover, or davinci), what website it comes from, and who it was annotated by.

Files beginning with 2. contain code for the analyses performed on the data collected from the annotated documents.

I) avg_COH_count.py produces a graph of the average per-document count for each coherence relation type, comparing
grover and human numbers.
II) avg_COH_length.py produces a graph comparing the average length of coherence relations between grover and
human documents.
III) avg_quality.py produces a graph of the average quality measures across grover and human
documents.
IV) COH_vs_nar_and_arg.py contains functions that calculate correlations between the number of each
coherence relation type and narrative/ argument presence as well as narrative/ argument quality measures.
V) SE_vs_nar_and_arg.py does the same for situation entities.
VI) human_ai_corres.py calculates the percent of documents in which the discourse modes present in an original
document also appear in the corresponding grover or davinci document.
VII) mode_type_analysis.py breaks down the number of documents that contain just narrative, just argument, both
narrative and argument, and no narrative or argument.
VIII) nonsensical_COH.py produces a graph with the proportion of nonsensical relations for each coherence relation type.
IX) number_of_COH_analysis.py calculates the correlation between the number of coherence relations and narrative/
argument presence as well as narrative/ argument quality measures.
X) clean_up_SE_coh.py, extract_annotations.py, and get-pip.py provide support code for these analyses.

Files beginning with 3. contain old analysis code that was not used in the final evaluation. See readme file within
relevant folder.

Folders and files beginning with 4. are related to corpus development.

I) 100_coh_disagreements and 100_se_disagreements list 100 examples of the most common coherence relation and situation
entity pairs that the annotators disagreed on.
II) confusion_matrices visualize the frequency of disagreements for all label pairs.
III) agreement_coh.py and agreement_SE.py contain code that calculate inter-rater agreement for coherence relations and
situation entities respectively.
IV) krippendorff_alpha.py is used in the agreement calculation.
