# code to change the line numbers in corresponding -annotation document when
# the segmentation in the original document was changed.

# TODO later: also deal with joining together two segments.
# (add command line T/F parameter to determine mode)

import sys

args = sys.argv
annotation_file = str(args[1])
segment_modified = int(args[2])
split_or_joined = str(args[3])

if split_or_joined not in ['split', 'joined']:
    raise Exception('invalid input')

# if mode is 'split', then segment_modified represents the segment that was split
# add 1 to all of the line numbers GREATER than segment_modified.
# if EQUAL to segment_modified, 
#  - keep the same if it's the beginning of a group,
#  - but add 1 if it's at the end of a group. 

# helper that takes a given line and increments appropriate numbers
def increment_helper(line):
    line_list = line.split()
    for i in range(len(line_list)):
        if line_list[i] > segment_modified:
            pass
            #replace with incremented version
        elif line_list[i] == segment_modified:
            pass
            #check whether it's at beginning or end of group

    return ' '.join(line_list)


# iterate through the annotation file and increment appropriate numbers
with open(annotation_file, 'r', encoding="utf-8") as annotations:
    lines = annotations.readlines()

with open(annotation_file, 'w', encoding="utf-8") as annotations:
    for line in lines:
        if line.strip() != "":
            annotations.write(increment_helper(line))
        else:
            annotations.write(line)