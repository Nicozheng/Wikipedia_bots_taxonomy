import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


## functions

def is_bot_page(title, botname):
    arg = "[User talk\:|User\:]" + botname
    if re.findall(arg, title):
        return 1
    else:
        return 0

def alter_bot_edit_category(bot_edit):
    """transfer botedit in its own user page as Wikipedia namespace"""
    new_category = []
    for title, botname, category in bot_edit[['title','username', 'category']].values:
        title = str(title)
        botname =str(botname)
        if is_bot_page(title, botname):
            new_category.append('wikipedia')
        else:
            new_category.append(category)
    return new_category

def load_dict(filepath):
    """load text file into dictionary"""
    with open(filepath) as f:
        result = f.readlines()
    result = "".join([c.strip() for c in result])
    result = result.replace("'", "\"")  # change format for json to identify
    result = json.loads(result)  ## change to dict format
    return result

tagging = lambda x,reg: 1 if re.findall(reg, x) else 0

def preprocessing(string_data):              # pre-processing string
    """pre-processing, return word set reduce stopwords
    """
    string_data = re.sub('[^a-zA-Z\-]', " ", string_data)
    string_data = re.sub(' +', " ", string_data)
    return string_data.lower()

## load bot data
bot_data = pd.read_csv("../data/Wikipedia_registered_bots_infor.txt", sep="\t")  ## registered bots information
bot_label = bot_label = pd.read_csv("../data/bot_taxonomy_train.txt", sep="\t")  ## annotation set
# bot_edit = pd.read_csv("/Users/Nico/GoogleDrive/data/bots/Wikipedia_botedits_all.txt", sep="\t")  ## bot edit histroy samples
bot_label.fillna(0, inplace=True)
bot_data['all_corpus'] = bot_data['userpage_content'].apply(str) + "@@@@" + bot_data['approval_history'].apply(str)
roles = bot_label.columns[2:].tolist()
print("there are %s bots." %len(bot_data))


## generate bot edit summary
bot_edit['category'] = alter_bot_edit_category(bot_edit)
botsummary = bot_edit.pivot_table(index='username', columns='category', values='revid', aggfunc='count').fillna(0)
activity_sum  = botsummary.apply(sum, axis=1)
botsummary = botsummary.apply(lambda x: x/activity_sum)
botsummary.reset_index(inplace=True)
# print(botsummary.head())


## generate verb features
### load verbs
key_verbs = load_dict("../data/verb_features.txt")
print("human filterd verbs: ")
# print(key_verbs)
verbs = set()
for i,j in key_verbs.items():
    verbs = verbs | set(j.split("|"))
print("there are total %s verbs" %len(verbs))

### tag document
vect = CountVectorizer(vocabulary=verbs,lowercase=True, binary=True).fit(bot_data['all_corpus'])
vect_matrix = vect.transform(bot_data['all_corpus']).toarray()
verb_features = vect.get_feature_names()
tagged_verb = pd.DataFrame(vect_matrix, columns=["V_%s" %c for c in range(len(verbs))])
tagged_verb['botname'] = bot_data['botname']

### generate feature importance dataset
test = pd.merge(bot_label, tagged_verb, on='botname', how='left')
x = test[["V_%s" %c for c in range(len(verbs))]]
feature_importance = {}
for i in roles:
    y = test[i]
    clf = RandomForestClassifier(n_estimators=10, random_state=22)
    clf.fit(x, y)
    tmp = pd.Series(dict(zip(verb_features, clf.feature_importances_)))
    feature_importance[i] = tmp
feature_importance = pd.DataFrame(feature_importance)
feature_importance.to_csv("../data/verb_feature_importance.txt", sep="\t")
plt.figure(figsize=(20,5))
sns.heatmap(feature_importance.T, cmap='YlGnBu')
plt.savefig("verb_feature_importance.png", bbox_inches='tight', dpi=300)

### load lexicon
lex = load_dict("../data/lexicon_features.txt")
## tagging
result_lex = {}
for key, reg in lex.items():
    result_lex[key] = bot_data['all_corpus'].apply(lambda x: tagging(preprocessing(x), reg))
result_lex = pd.DataFrame(result_lex)
result_lex['botname'] = bot_data['botname']
# print(result_lex.head())


## merge all features
botsummary.columns = ['botname', 'content', 'content-talk', 'infrastructure', 'user', 'user-talk', 'wikipedia']
training_table = pd.merge(botsummary, tagged_verb, how='left')
training_table = pd.merge(training_table, result_lex, how='left')
training_table.to_csv("../data/bot_taxonomy_all_features.txt", sep="\t", index=None)
