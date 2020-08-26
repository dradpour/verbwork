#Authors: Vinay Ashokkumar


'''import json
with open('yelp_academic_dataset_review.json') as data:
    js = json.load(data)
'''

import pickle
import json

dataset = { 1 : [], 2 : [], 3 : [], 4 : [], 5 : [] }

reviews_count = 0
required_count=1000
f = open('yadr.json')
#objects = ijson.parse(f)
for line in f:
    review = json.loads(line)
    if(len(dataset[review['stars']])<required_count):
        dataset[review['stars']].append(review)
        reviews_count = reviews_count+1
    if(reviews_count==5*required_count):
        break

pickle.dump(dataset, open('dataset.p','wb'))

'''for review in dataset[1]:
    print(review['stars'])
    print()
'''
