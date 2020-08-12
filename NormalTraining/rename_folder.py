import json
import os
import re

directory = "../datasets/Stamp_Recognition/To_Use_Datasets/test"
with open('author_2.json') as f:
    data = json.load(f)
list_author = next(os.walk(directory))[1]


for author in list_author:
    author_lower = author.lower()
    print(author)
    for each_author in data.keys():
        author_name = re.sub(r'\W+', '', each_author)
        if author_lower == author_name.lower():
            old_name = os.path.join(directory, author)
            new_name = os.path.join(directory, str(data[each_author]))
            os.rename(old_name, new_name)
            # print("author: ", author, " code: ", data[each_author])

