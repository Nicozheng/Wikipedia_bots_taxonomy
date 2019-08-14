import pandas as pd
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

## functions
def preprocessing(string_data):              # pre-processing string
    """pre-processing, return word set reduce stopwords
    """
    string_data = re.sub('[^a-zA-Z\-]', " ", string_data)
    string_data = re.sub(' +', " ", string_data)
    return string_data.lower()

def extract_verbs(text_list):
    verbset = set()
    for text in text_list:
        doc = nlp(text)
        for token in doc:
            if token.tag_.startswith("VB"):
                verbset.add(token.text)
    return verbset


## load bot data
bot_data = pd.read_csv("../data/Wikipedia_registered_bots_infor.txt", sep="\t")  ## registered bots information
bot_label = bot_label = pd.read_csv("../data/bot_taxonomy_train.txt", sep="\t")  ## annotation set
bot_label.fillna(0, inplace=True)
print("there are %s bots." %len(bot_data))

bot_data['all_corpus'] = bot_data['userpage_content'].apply(str) + "@@@@" + bot_data['approval_history'].apply(str)
nlp = spacy.load('en')
verbset = extract_verbs(bot_data['all_corpus'])
print("identified %s verbs" %len(verbset))

## tag documents
vect = CountVectorizer(vocabulary=verbset,lowercase=True,binary=True).fit(bot_data['all_corpus'])
vect_matrix = vect.transform(bot_data['all_corpus']).toarray()
word_freq = pd.Series(dict(zip(vect.get_feature_names(), np.sum(vect_matrix, axis=0))))
extracted_verbs = word_freq[word_freq >= 100].sort_values(ascending=False).index.tolist()  ## extract verbs occured in more than 100 documents
print("extract verbs occured in more than 100 bots userpage/approval_history")

## retag documents to apply random forest model
vect = CountVectorizer(vocabulary=extracted_verbs,lowercase=True).fit(bot_data['all_corpus'])
vect_matrix = vect.transform(bot_data['all_corpus']).toarray()
verb_features = vect.get_feature_names()
bot_verbs = pd.DataFrame(vect_matrix, columns=["V%s" %c for c in range(len(extracted_verbs))])
bot_verbs['botname'] = bot_data['botname']

## apply RandomForestClassifier to filter verbs have feature importance larger than 0.005
roles = bot_label.columns[2:].tolist()
test = pd.merge(bot_label, bot_verbs, on='botname', how='left')
x = test[["V%s" %c for c in range(len(extracted_verbs))]]
key_verbs = {}
for i in roles:
    y = test[i]
    clf = RandomForestClassifier()
    clf.fit(x, y)
    tmp = pd.Series(dict(zip(verb_features, clf.feature_importances_))).sort_values(ascending=False)
    key_verbs[i] = "%s" %"|".join(tmp[tmp>0.005].index)

## print machine filtered verbs
with open('../data/machine_filtered_verbs.txt', 'w') as f:
    print(str(key_verbs), file=f)
