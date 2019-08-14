import pandas as pd
import numpy as np
from skmultilearn.adapt import MLkNN, BRkNNaClassifier, MLARAM
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# load bot data
training_table = pd.read_csv("../data/bot_taxonomy_all_features.txt", sep="\t")  # feature set
bot_label = pd.read_csv("../data/bot_taxonomy_train.txt", sep="\t")  ## annotation set
bot_label.fillna(0, inplace=True)
roles = bot_label.columns[2:].tolist()
aggregated_data = pd.merge(bot_label, training_table, how='left')
aggregated_data = aggregated_data.dropna()


# feature categories
edit_features = ['content', 'content-talk', 'infrastructure', 'user', 'user-talk', 'wikipedia']  # edit frequency features
verb_features = ["V_%s" %c for c in range(174)]  # frequent verb features
lex_features = ['lex_sources', 'lex_policies']  # lexicon features

orders = {'Edits' : edit_features, "Edits + Verbs" : edit_features + verb_features, "Edits + Verbs + Lex" : edit_features + verb_features + lex_features}

for feature, value in orders.items():
    ## check training features
    x, y = aggregated_data[value].values, aggregated_data[roles].values
    print("Using Feature Set: %s" %feature)
    print("Shape of X", np.shape(x))
    print("Shape of Y", np.shape(y))

    ## classifiers
    classifiers = {'BR': BRkNNaClassifier(k=5), 'MLkNN': MLkNN(k=5), 'MLARAM': MLARAM()}

    ## 10 folder cross validation
    results = {}   # record macro and micro F1 each round
    individual_results = {}  # record precision, recall, and F1 for each role
    for model, clf in classifiers.items():
        folders = KFold(n_splits=10, random_state=1002).split(x, y)
        tmp = {}
        tmp2 = {}
        i = 0
        for train,test in folders:
            clf.fit(x[train], y[train])
            y_pred = clf.predict(x[test])
            if not isinstance(y_pred, np.ndarray):
                y_pred = y_pred.toarray()
    #         print(y_test)
            tmp[i] = {"macro-f1": f1_score(y[test],y_pred, average='macro'),
                         "micro-f1": f1_score(y[test],y_pred, average='micro')}
            ## F1 for each class
            individual_role = {}
            for col, name in enumerate(roles):
                individual_role[name] = {"Precision": precision_score(y[test][:,col], y_pred[:,col]),
                                         "Recall": recall_score(y[test][:,col], y_pred[:,col]),
                                        "F1 Score": f1_score(y[test][:,col], y_pred[:,col])}
            tmp2[i] = individual_role
            i += 1
        results[model] = tmp
        individual_results[model] = tmp2
    for model, scores in results.items():
        print(model)
        print(pd.DataFrame(scores).T.mean())


## check MLARAM's precision and recall for each role
result_mlaram = []
for i in range(10):
    tmp = pd.DataFrame(individual_results['MLARAM'][i])
    tmp['iter'] = i
    result_mlaram.append(tmp)
result_mlaram = pd.concat(result_mlaram)
result_mlaram = result_mlaram.reset_index()
summary = result_mlaram.groupby('index').agg('mean')
del summary['iter']
summary.T.plot.bar(figsize=(12,6), rot=0, fontsize=12)
plt.savefig("MLARAM_individual_role_score.png", bbox_inches='tight', dpi=300)

## predict
# print(training_table.columns)
x = training_table[edit_features + verb_features + lex_features].values
y_pred = classifiers['MLARAM'].predict(x)
bot_label_pred = pd.DataFrame(y_pred, columns=roles)
bot_label_pred['botname'] = training_table['botname']
bot_label_pred.to_csv("../data/bot_taxonomy_predict.txt", index=None, sep="\t")
