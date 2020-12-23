### function that takes an old_dict and a clean_dict and loads old into clean
### clean_dict should have keys defined already for each SE type
def clean_up_SE_types(old_dict, clean_dict):
    for SE_type in old_dict.keys():
        if SE_type in clean_dict.keys():
            clean_dict[SE_type] += old_dict[SE_type]
        elif SE_type in ['BOUNDED EVENT (SPECIFIC)', 'BOUNDED EVENT (GENERIC)', 'BOUNDED EVENT (SPECIFIC_']:
            clean_dict['BOUNDED EVENT'] += old_dict[SE_type]
        elif SE_type in ['UNBOUNDED EVENT (SPECIFIC)', 'UNBOUNDED EVENT (GENERIC)', 'UNBOUNDED EVENT (GENERIC0', 'NBOUNDED EVENT (SPECIFIC)']:
            clean_dict['UNBOUNDED EVENT'] += old_dict[SE_type]
        elif SE_type in ['COERCED STATE (SPECIFIC)', 'COERCED STATE (GENERIC)', 'COERCED STATE (STATIC)']:
            clean_dict['COERCED STATE'] += old_dict[SE_type]
        elif SE_type in ['PERFECT COERCED STATE (SPECIFIC)', 'PERFECT COERCED STATE (GENERIC)']:
            clean_dict['PERFECT COERCED STATE'] += old_dict[SE_type]
        elif SE_type in ['GENERIC SENTENCE (STATIC)', 'GENERIC SENTENCE (SATIC)', 'GNERIC SENTENCE (STATIC)', 'GNERIC SENTENCE (STATIC0', 'GENERIC SENTENCE (STATAIC)', 'GENERIC STATE (STATIC)', 'GENERIC SENTENCE (DYNAMIC)', 'GENERIC SENTENCE (HABITUAL)']:
            clean_dict['GENERIC SENTENCE'] += old_dict[SE_type]
        elif SE_type in ['GENERALIZING SENTENCE (DYNAMIC)', 'GENERALIZING SENTENCE (STATIVE)']:
            clean_dict['GENERALIZING SENTENCE'] += old_dict[SE_type]
        elif SE_type in ['BASIC SENTENCE', 'BAISC STATE', 'BASIC STATEA', 'BASIC SETATE']:
            clean_dict['BASIC STATE'] += old_dict[SE_type]
        elif SE_type in ['OHTER', 'ITGER', 'OTEHR', 'OTHE R']:
            clean_dict['OTHER'] += old_dict[SE_type]

    for k in clean_dict.keys():
        clean_dict[k] /= sum(old_dict.values())


### Clean up and rename coherence relations 
### clean_dict should have keys defined already for each Coh relation
def clean_up_coh_rels(old_dict, clean_dict):
    for relation in old_dict.keys():
        if relation in clean_dict.keys():
            clean_dict[relation] += old_dict[relation]
        elif relation in ['ce', 'cex', 'cew', 'cer']:
            clean_dict['Cause/effect'] += old_dict[relation]
        elif relation in ['elab', 'elabx', 'ealb', 'elav', 'elabl']:
            clean_dict['Elaboration'] += old_dict[relation]
        elif relation in ['same', 'samex']:
            clean_dict['Same'] += old_dict[relation]
        elif relation in ['attr', 'attrx', 'attrm']:
            clean_dict['Attribution'] += old_dict[relation]
        elif relation in ['deg', 'degenerate', 'mal']:
            clean_dict['Degenerate'] += old_dict[relation]
        elif relation in ['sim', 'simx']:
            clean_dict['Similarity'] += old_dict[relation]
        elif relation in ['contr', 'contrx']:
            clean_dict['Contrast'] += old_dict[relation]
        elif relation in ['temp', 'tempx']:
            clean_dict['Temporal sequence'] += old_dict[relation]
        elif relation in ['ve', 'vex']:
            clean_dict['Violated expectation'] += old_dict[relation]
        elif relation in ['examp', 'exampx']:
            clean_dict['Example'] += old_dict[relation]
        elif relation in ['cond','condx']:
            clean_dict['Condition'] += old_dict[relation]
        elif relation in ['gen', 'genx']:
            clean_dict['Generalization'] += old_dict[relation]
        elif relation in ['rep']:
            clean_dict['Repetition'] += old_dict[relation]