import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
import time
from auxilaryFunc import *


#Trains a classifier for dissimilarity, already has 3 built in classes
def train_clf(ratios, edgeThresList, clfName, verbose=True, trainOnly = True, classType='log', XYReps =XYRepsFolder):
    if verbose:
        print("Training clf")
        t = time.time()
    if trainOnly:
        trainEvalTestDirs = ['train']
    else:
        trainEvalTestDirs = ['train','val']
    for trainEvalTestDir in trainEvalTestDirs:
        imageDir = os.sep.join([databaseDir, 'images', trainEvalTestDir])
        imagesNames = os.listdir(imageDir)
        imagesNames = list(filter(lambda name: name.endswith('.jpg'), imagesNames))

        X_tot = None
        Yb_tot = None
        for name in imagesNames:
            name = name[:-4]
            for edgeThres in edgeThresList:
                xyName = os.sep.join(
                    [XYReps, name + "_XY_"  + str(edgeThres) + ".p"])
                try:
                    (X, Yb) = pickle.load(open(xyName, "rb"))
                except:
                    continue
                if len(Yb)==0:
                    continue
                if X_tot is None:
                    X_tot = X
                    Yb_tot = Yb
                else:
                    X_tot.extend(X)
                    Yb_tot.extend(Yb)

    Y_tot = [(np.array(y)<0.6).astype(int) for y in Yb_tot]
    Y = np.array([np.mean(y) for y in Y_tot])
    Y = (Y <= 0.9).astype(int)
    X = np.array([getFeaturesFromMyX(x, ratios) for x in X_tot])
    if classType=='lr':
        clf1 = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', max_iter=200,verbose=False).fit(X, Y)
    elif classType=='mlp':
        np.seterr(all='ignore')
        clf1 = MLPClassifier(random_state=0,verbose=False,learning_rate_init=0.001,max_iter=2000, early_stopping=True, activation='logistic', alpha=0.001, hidden_layer_sizes=(96,)*5).fit(X, Y)
        np.seterr(all='raise')
    elif classType=='for':
        clf1 = RandomForestRegressor(random_state=0, verbose=False, n_estimators=20, max_depth=15, min_samples_leaf=400, min_samples_split=0.05).fit(X, Y)
    elif classType=='svm':
        clf1 = SVC(random_state=0, C=1000, gamma='scale', probability=True, max_iter=10000).fit(X, Y)
    else:
        raise ("error")
    pickle.dump(clf1, open(clfName, "wb"))
    if verbose:
        print("Finished training clf. Time taken: " + str(time.time()-t) )
