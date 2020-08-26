#Authors: Vinay Ashokkumar

#import json
import pickle
#Define Punctual and Keyword based features to which sarcasm traits are defined upon
Key =  ['WOW', 'OMG', 'HAHA',   'DAMN', 'GOD']
Punct = ['?', '()', '!', '..', '""', ',', '.']
Self_Reference=['I',	'ME',	'MY',	'MINE',	'MYSELF', 'WE', 'US', 'THEY', 'OURSELVES'] #Fill with 1st person pronouns
#Initilize the feature dictionary
dictionary = {'wow': None, 'WOW': None, 'Wow': None, 'OMG': None, 'omg': None,'Omg': None,'HAHA': None, 'haha': None, 'Haha': None, 'damn': None, 'Damn': None, 'DAMN': None, 'GOD': None, 'God': None, 'god': None, '?': None, '()': None, '!': None, '..': None, '""': None, ',': None, '.': None, 'Superlatives': None, 'Self-Referenced': None, 'Spring': None, 'Fall': None, 'Summer': None, 'Winter': None} 
superlatives = ['LEAST', 'MOST', 'WORST']  
neuralnetworkinput=[]
#read json file into python
#with open('C:\Users\Vinay\Desktop\CSE 674\yelp data\yelp_academic_dataset_review.json') as json_data:
   # d = json.load(json_data)
    #print(d)

#Unload Review Data Set pickle file for preprocessing 
dataset=pickle.load(open("C:\Users\Vinay\Desktop\CSE 674\yelp data/dataset.p", "rb"))
#print(dict(dataset[1][1])['date'].split('-')[1])


#Define method to find no. of Keyword based features in corpus
def Keyword(text):
    text = text.upper().split()
    Keyword_dict={i:text.count(i) for i in Key}
    return Keyword_dict 
    

#print Keyword('Bert and Ernie god')
#Define method to find no. of Punctual based features in corpus

def Punctuation(text):
    text = list(text)
    Punctuation_dict={i:text.count(i) for i in Punct}
    return Punctuation_dict 

#nltk.download('averaged_perceptron_tagger')
#print Punctuation('Chuck, and Harry!')


#Function to identify the type of referentilaity used in text  
def Referentiality(text):
    text = text.upper().split()
    s=0
    for i in Self_Reference:
        s=s+text.count(i) 
        return s
    
#print Referentiality('I we may she they Vinay us this that')

#Function to identify the seasons when the review has been entered.
def Season(month):
    if month in [11,12,1]:
        return '0'#Return WINTER months
    
    elif month in [2,3,4]:
        return '1'#Return SPRING months
        
    elif month in [5,6,7,8]:
        return '2'#Return SUMMER months
        
    elif month in [9,10]:
        return '3'#Return FALL months
        
    else:
        return 'No Season'
        
#print Season(12)

 
def Superlatives(text):
    s=0
    text=text.upper().split()
    for i in text:
        if(i.endswith('EST') or (i in superlatives)):
            s=s+1
        
    return s       
        
    
#print Superlatives('X is the most awesomest person.')   


roi = 5
count = 0
#Iterate all star reviews seperately to create the cummulative dictionary              
for review in dataset[roi]:
    review = dict(review)
    print
    print review['text']
    print
    label = input ('Enter label : ')
    while(label!=0 and label!=1):
            label = input ('Enter label again: ')        
    result_dict={'label' : label}
    result_dict.update(Keyword(review['text']))
    result_dict.update(Punctuation(review['text']))
    result_dict['Superlative_count']=Superlatives(review['text'])
    result_dict['Self_Referentiality_count']=Referentiality(review['text']) 
    result_dict['Season']=Season(int(review['date'].split('-')[1]))
    print result_dict
    print
    print
    print
    print '--------------------------------------------------------------------------------------------'
    #result_dict['Sarcasm']=0
    neuralnetworkinput.append(result_dict)
    count = count+1
    if count==300:
        break
    
pickle.dump(neuralnetworkinput, open('C:\Users\Vinay\Desktop\CSE 674\yelp data\\'+str(roi)+'star_frames.p','wb'))  
print('Pickled to glory..')
'''
test = pickle.load(open('C:\Users\Vinay\Desktop\CSE 674\yelp data\\5star_frames.p','rb'))
print('TEST')
print(test)  
'''

'''for x in neuralnetworkinput:
    print x
    print '\n'      
'''
'''p_one=0
p_two=0
p_three=0
p_four=0
p_five=0
p_six=0
p_seven=0
p_eight=0
p_nine=0
p_ten=0
p_eleven=0
p_tweleve=0
p_thirteen=0
p_fourteen=0
p_fifteen=0



if('wow' in d):
    p_one=p_one+1
    
elif('Wow' in d):
    p_two=p_two+1

elif('WOW' in d):
    p_three=p_three+1

elif('OMG' in d):
    p_four=p_four+1
    
elif('omg' in d):
    p_five=p_five+1
    
elif('Omg' in d):
    p_six=p_six+1
    
elif('HAHA' in d):
    p_seven=p_seven+1
    
elif('haha' in d):
    p_eight=p_eight+1
    
elif('Haha' in d):
    p_nine=p_nine+1
    
elif('damn' in d):
    p_ten=p_ten+1
    
elif('Damn' in d):
    p_eleven=p_eleven+1
    
elif('DAMN' in d):
    p_twelve=p_twelve+1
    
elif('GOD' in d):
    p_thirteen=p_thirteen+1
    
elif('God' in d):
    p_fourteen=p_fourteen+1
    
elif('god' in d):
    p_fifteen=p_fifteen+1
    
    

k_one=0
k_two=0
k_three=0
k_four=0
k_five=0
k_six=0
k_seven=0


if('?' in d):
    k_one=k_one+1
    
elif('()' in d):
    k_two=k_two+1

elif('!' in d):
    k_three=k_three+1

elif('..' in d):
    k_four=k_four+1
    
elif('""' in d):
    k_five=k_five+1
    
elif(',' in d):
    k_six=k_six+1
    
elif('.' in d):
    k_seven=k_seven+1
'''
    
