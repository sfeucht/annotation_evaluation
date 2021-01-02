import os
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"


valid_full_SE_types = ['BOUNDED EVENT (SPECIFIC)', 'BOUNDED EVENT (GENERIC)', 
'UNBOUNDED EVENT (SPECIFIC)', 'UNBOUNDED EVENT (GENERIC)', 'BASIC STATE', 'COERCED STATE (SPECIFIC)',
'COERCED STATE (GENERIC)', 'PERFECT COERCED STATE (SPECIFIC)', 'PERFECT COERCED STATE (GENERIC)', 
'GENERIC SENTENCE (STATIC)', 'GENERIC SENTENCE (DYNAMIC)', 'GENERIC SENTENCE (HABITUAL)', 
'GENERALIZING SENTENCE (DYNAMIC)', 'GENERALIZING SENTENCE (STATIVE)', 'QUESTION', 'IMPERATIVE', 
'OTHER', 'NONSENSE']
annotators = {"Sheridan":[],"Muskaan":[],"Kate":[]}

### function that takes in an SE type and converts to non-typo version if necessary
### if the typo has not been accounted for yet, return the typo version and print a message.
def fix_SE_typo(old, f, annotator):
    if old in ['BOUNDED EVENT (SPECIFIC_', 'BOUNDED EVENT (SPECIFIC)_', 'BOUNDED EVENT (SPECIFIC0']:
        return 'BOUNDED EVENT (SPECIFIC)'
    elif old in ['NBOUNDED EVENT (SPECIFIC)']:
        return 'UNBOUNDED EVENT (SPECIFIC)'
    elif old in ['UNBOUNDED EVENT (GENERIC0', 'UNBOUNDED EVNET (GENERIC)']:
        return 'UNBOUNDED EVENT (GENERIC)'
    elif old in ['COERCED STATE (SPECIFC)']:
        return 'COERCED STATE (SPECIFIC)'
    elif old in ['PERFECT COERCED STAE (GENERIC)', 'PERECT COERCED STATE (GENERIC)']:
        return 'PERFECT COERCED STATE (GENERIC)'
    elif old in ['GENERIC SENTENCE (SATIC)', 'GNERIC SENTENCE (STATIC)', 'GNERIC SENTENCE (STATIC0', 'GENERIC SENTENCE (STATAIC)', 'GENERIC STATE (STATIC)', 'GENERIC SENTENCE (STATIC0',]:
        return 'GENERIC SENTENCE (STATIC)'
    elif old in ['GENERALIING SENTENCE (DYNAMIC)']:
        return 'GENERALIZING SENTENCE (DYNAMIC)'
    elif old in ['BASIC SENTENCE', 'BAISC STATE', 'BASIC STATEA', 'BASIC SETATE', 'BASIC STATE A', 'BASICE STATE']:
        return 'BASIC STATE'
    elif old in ['OHTER', 'ITGER', 'OTEHR', 'OTHE R']:
        return 'OTHER'
    elif old[:5] == 'OTHER':
        return old
    else:
        if old not in valid_full_SE_types:
            if old.upper() not in valid_full_SE_types:
                print(annotator, f, "typo: ", old)
        return old.upper()


### this code fixes SE typos in the whole corpus
### if a typo has been seen before and is in this code, fixes it, otherwise prints
for annotator in annotators:
    folder = os.listdir("{}/".format(annotator))
    SE_files = [f for f in folder if 'annotation' not in f]
    for f in SE_files:
        with open(path+("{}/".format(annotator))+f,'r', encoding="utf-8") as annotated_doc:
            lines = annotated_doc.readlines()
        
        with open(path+("{}/".format(annotator))+f,'w', encoding="utf-8") as annotated_doc:
            for line in lines:
                if line.strip() != "" and "***" not in line:
                        try:
                            s = line.strip().split('##')[1].split('//')[0]
                            s = s.strip('\* ')
                            new_line = line.replace(s, fix_SE_typo(s, f, annotator))
                            annotated_doc.write(new_line)
                        except:
                            annotated_doc.write(line)
                else:
                    annotated_doc.write(line)


valid_full_coh_rels = ['ce', 'cex', 'elab', 'elabx', 'same', 'samex', 'attr', 
'attrx', 'attrm', 'attrmx', 'deg', 'sim', 'simx', 'contr', 'contrx', 'temp',
'tempx', 've', 'vex', 'examp', 'exampx', 'cond', 'condx', 'gen', 'genx', 'rep']

### assumes that typos are for non-x versions. 
def fix_coh_typo(old, f, annotator):
    if old in valid_full_coh_rels: 
        return old
    elif old in ['cew', 'cer']:
        return 'ce'
    elif old in ['ealb', 'elav', 'elabl', 'elb', 'felab']:
        return 'elab'
    elif old in ['degenerate', 'mal']:
        return 'deg'
    else:
        if old not in valid_full_coh_rels:
            if old.lower() not in valid_full_coh_rels:
                print(annotator, f, "typo: ", old)
        return old.lower()

### this code fixes coherence typos in the whole corpus
### if a typo has been seen before and is in this code, fixes it, otherwise prints
for annotator in annotators:
    folder = os.listdir("{}/".format(annotator))
    coh_files = [f for f in folder if 'annotation' in f]
    for f in coh_files:
        with open(path+("{}/".format(annotator))+f,'r', encoding="utf-8") as annotations:
            lines = annotations.readlines()
        
        with open(path+("{}/".format(annotator))+f,'w', encoding="utf-8") as annotations:
            for line in lines:
                if line.strip() != "":
                        try:
                            tag = line.strip().split('//')[0].split()[-1]
                            new_line = line.replace(tag, fix_coh_typo(tag, f, annotator))
                            annotations.write(new_line)
                        except:
                            annotations.write(line)
                else:
                    annotations.write(line)

################################################################################

valid_simplified_SE_types = ['BOUNDED EVENT', 'UNBOUNDED EVENT', 'BASIC STATE', 'COERCED STATE',
'PERFECT COERCED STATE', 'GENERIC SENTENCE', 'GENERALIZING SENTENCE', 'QUESTION',
'IMPERATIVE', 'OTHER', 'NONSENSE']

### takes in a single SE type and converts to one of the valid SE types
### collapses BOUNDED EVENT (SPECIFIC) and BOUNDED EVENT (GENERIC) into a single category
### ASSUMES that typos have been corrected already!
def simplify_SE_type(old):
    old = old.upper()
    if old in valid_simplified_SE_types:
        return old
    elif old in ['BOUNDED EVENT (SPECIFIC)', 'BOUNDED EVENT (GENERIC)']:
        return 'BOUNDED EVENT'
    elif old in ['UNBOUNDED EVENT (SPECIFIC)', 'UNBOUNDED EVENT (GENERIC)']:
        return 'UNBOUNDED EVENT'
    elif old in ['COERCED STATE (SPECIFIC)', 'COERCED STATE (GENERIC)']:
        return 'COERCED STATE'
    elif old in ['PERFECT COERCED STATE (SPECIFIC)', 'PERFECT COERCED STATE (GENERIC)']:
        return 'PERFECT COERCED STATE'
    elif old in ['GENERIC SENTENCE (STATIC)', 'GENERIC SENTENCE (DYNAMIC)', 'GENERIC SENTENCE (HABITUAL)']:
        return 'GENERIC SENTENCE'
    elif old in ['GENERALIZING SENTENCE (DYNAMIC)', 'GENERALIZING SENTENCE (STATIVE)']:
        return 'GENERALIZING SENTENCE'
    elif old[:5] in ['OTHER']:
        return 'OTHER'

### function that takes an old_dict and a clean_dict and loads old into clean
### simplifies the categories, i.e. just BOUNDED EVENT or COERCED STATE
### also normalizes counts in clean_dict by number of values in old_dict
### clean_dict should have keys defined already for each SE type
def simplify_all_SE_types(old_dict, clean_dict):
    for SE_type in old_dict.keys():
        clean_dict[simplify_SE_type(SE_type)] += old_dict[SE_type]
    for k in clean_dict.keys():
        clean_dict[k] /= sum(old_dict.values())


################################################################################

valid_coh_rels = ['Cause/effect', 'Elaboration', 'Same', 'Attribution', 'Degenerate',
'Similarity', 'Contrast', 'Temporal sequence', 'Violated expectation', 'Example',
'Condition', 'Generalization', 'Repetition']

### takes in a single coh relation and converts to one of the valid relations
### TODO: assumes that typos are taken care of
def clean_version_coh(old):
    if old in valid_coh_rels: #this shouldn't happen but adding for completeness
        return old
    elif old in ['ce', 'cex']:
        return 'Cause/effect'
    elif old in ['elab', 'elabx']:
        return 'Elaboration'
    elif old in ['same', 'samex']:
        return 'Same'
    elif old in ['attr', 'attrx', 'attrm']:
        return 'Attribution'
    elif old in ['deg']:
        return 'Degenerate'
    elif old in ['sim', 'simx']:
        return 'Similarity'
    elif old in ['contr', 'contrx']:
        return 'Contrast'
    elif old in ['temp', 'tempx']:
        return 'Temporal sequence'
    elif old in ['ve', 'vex']:
        return 'Violated expectation'
    elif old in ['examp', 'exampx']:
        return 'Example'
    elif old in ['cond','condx']:
        return 'Condition'
    elif old in ['gen', 'genx']:
        return 'Generalization'
    elif old in ['rep']:
        return 'Repetition'

### Clean up and rename coherence relations 
### clean_dict should have keys defined already for each Coh relation
def clean_up_coh_rels(old_dict, clean_dict):
    for relation in old_dict.keys():
        print(relation, clean_version_coh(relation))
        clean_dict[clean_version_coh(relation)] += old_dict[relation]