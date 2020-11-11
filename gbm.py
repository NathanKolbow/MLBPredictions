import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as gb_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

df = pd.read_csv("pitches.csv")

to_pop = ["Unnamed: 0", "pitch_type", "stand", "p_throws", "inning_topbot", "pitcher_name", "catcher_name", "umpire_name", "batter_name", "game_date"]
for item in to_pop:
    df.pop(item)

y = df.pop("description").values
y_lookup, y = np.unique(y, return_inverse = True)
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,stratify=y)

subs = [.1, .3, .5, .7, .9, 1]
sub_gbms = []
sub_scores = []
for sub in subs:
    sub_gbms.append(gb_classifier(subsample=sub,learning_rate=.1,max_depth=25,n_estimators=100))

for gbm in sub_gbms:
    gbm.fit(X_train,y_train)
    sub_scores.append(gbm.score(X_test, y_test))
    
best_sub_arg = np.argmax(np.array(sub_scores))
best_sub = subs[best_sub_arg]
print(sub_scores)

depths = [5, 10, 25, 50, 100, 200]
depth_gbms = []
depth_scores = []
for depth in depths:
    depth_gbms.append(gb_classifier(subsample=best_sub,learning_rate=.1,max_depth=depth,n_estimators=100))

for gbm in depth_gbms:
    gbm.fit(X_train,y_train)
    depth_scores.append(gbm.score(X_test, y_test))
    
best_depth_arg = np.argmax(np.array(depth_scores))
best_depth = depths[best_depth_arg]
print(depth_scores)
    
rates = [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9]
rate_gbms = []
rate_scores = []
for rate in rates:
    rate_gbms.append(gb_classifier(subsample=best_sub,learning_rate=rate,max_depth=best_depth,n_estimators=100))
    
for gbm in rate_gbms:
    gbm.fit(X_train,y_train)
    rate_scores.append(gbm.score(X_test, y_test))
    
best_rate_arg = np.argmax(np.array(rate_scores))
best_rate = rates[best_rate_arg]
print(rate_scores)
    
nests = [10, 100, 150, 300, 500, 1000, 10000]
nest_gbms = []
nest_scores = []
for nest in nests:
    nest_gbms.append(gb_classifier(subsample=best_sub,learning_rate=best_rate,max_depth=best_depth,n_estimators=nest))
    
for gbm in nest_gbms:
    gbm.fit(X_train,y_train)
    nest_scores.append(gbm.score(X_test, y_test))
    
best_nest_arg = np.argmax(np.array(nest_scores))
best_nest = nests[best_nest_arg]
print(nest_scores)

best_gbm = gb_classifier(learning_rate=best_rate,max_depth=best_depth,n_estimators=best_nest)
best_gbm.fit(X_train,y_train)
print("subsample =", best_sub, "learning_rate =", best_rate, "max_depth =", best_depth, "n_estimators =", best_nest, "score:", best_gbm.score(X_test, y_test))