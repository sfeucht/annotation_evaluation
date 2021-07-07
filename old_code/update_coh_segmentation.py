# code to change the line numbers in corresponding -annotation document when
# the segmentation in the original document was changed.

import sys
args = sys.argv
annotation_file = str(args[1])
segment_modified = int(args[2])
split_or_joined = str(args[3])


# helper that takes a given line and increments appropriate numbers (see 'split' comments)
def increment_helper(line):
    line_list = line.split()
    for i in range(len(line_list)):
        if i < 4 and line_list[i] != '?':
            if int(line_list[i]) > segment_modified:
                line_list[i] = str(int(line_list[i]) + 1)
            elif int(line_list[i]) == segment_modified:
                if i in [0, 2]: # if at beginning of group, don't change number
                    pass
                elif i in [1, 3]: # if at end of group, increment number
                    line_list[i] = str(int(line_list[i]) + 1)

    return ' '.join(line_list) + '\n'

# helper that takes a given line and decrements appropriate numbers (see 'joined' comments)
def decrement_helper(line):
    line_list = line.split()
    for i in range(len(line_list)):
        if i < 4 and line_list[i] != '?':
            if int(line_list[i]) > segment_modified:
                line_list[i] = str(int(line_list[i]) - 1)

    return ' '.join(line_list) + '\n'


# if mode is 'split', then segment_modified represents the segment that was split
# add 1 to all of the line numbers GREATER than segment_modified.
# if EQUAL to segment_modified, 
#  - keep the same if it's the beginning of a group,
#  - but add 1 if it's at the end of a group. 
if split_or_joined == 'split':
    # iterate through the annotation file 
    with open(annotation_file, 'r', encoding="utf-8") as annotations:
        lines = annotations.readlines()
    # increment appropriate numbers
    with open(annotation_file, 'w', encoding="utf-8") as annotations:
        for line in lines:
            if line.strip() != "":
                annotations.write(increment_helper(line))
            else:
                annotations.write(line)


# if mode is 'joined', then segment_modified represents the first segment.
# decrement every line number GREATER than segment_modified by one 
# if EQUAL to segment_modified, do not change
# may get some self referential annotations, but that's okay (e.g. 0 1 2 2 -> 0 1 1 1)
elif split_or_joined == 'joined':
    # iterate through the annotation file
    with open(annotation_file, 'r', encoding="utf-8") as annotations:
        lines = annotations.readlines()
    # decrement appropriate numbers
    with open(annotation_file, 'w', encoding="utf-8") as annotations:
        for line in lines:
            if line.strip() != "":
                annotations.write(decrement_helper(line))
            else:
                annotations.write(line)


else:
    raise Exception('invalid input. usage: segment_modified, split_or_joined')