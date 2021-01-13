import os

file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"

counter = 0
with open(path+("{}/".format('Kate'))+'30920174446.txt','r', encoding="utf-8") as annotated_doc:

                # initialize container for the SE types
                SE_temp_container = []
                for line in annotated_doc:

                    if line.strip() != "":

                        # save document-level ratings
                        if "***" not in line:
                            if '##' not in line:
                                print(line, 'no ##')
                            else:
                                counter += 1

print(counter)
                            