import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from skimage import color
import sklearn
if sklearn.__version__!= '0.20.3':
    import warnings
    warnings.warn('Using a different vesion of sklearn than 0.20.3. Isolation Forest might not work well.')
from sklearn.ensemble import IsolationForest


#Class designed for fast retrieval of data for segments
class VectorDataOrig:
    def __init__(self, overSeg, ratios, allRatioCnnRes, EdgeMapList, maxSize, Image, clfSplit = 4, ifFilter=True):
        overSegSmall = overSeg[1::4, 1::4]
        self.segVectors = [None]
        self.segClustersL2 = [None]
        numSegs = overSeg.max()
        self.EdgeMapList = EdgeMapList
        self.overSeg = overSegSmall
        self.allRatioCnnRes = []
        self.ratios = ratios
        self.maxSize = maxSize
        self.n_trees = 10
        self.clfList = [None]
        self.clfSplit = clfSplit
        self.segL = [None] * numSegs * 2
        self.segA = [None] * numSegs * 2
        self.segB = [None] * numSegs * 2
        self.lab=color.rgb2lab(Image)[::2,::2,:]
        self.ifFilter= ifFilter

        for idx in range(len(ratios)):
            allRatioCnnRes[idx] = allRatioCnnRes[idx]

        for cnnRes in allRatioCnnRes:
            temp = []
            for outLayer in cnnRes:
                temp.append( cv2.resize(outLayer, (overSegSmall.shape[1], overSegSmall.shape[0]),
                                         interpolation=cv2.INTER_LINEAR))
            self.allRatioCnnRes.append(temp)

        for _ in range(numSegs * 2):
            temp = []
            for _ in ratios:
                temp.append(list())
            self.segVectors.append(copy.deepcopy(temp))
            self.segClustersL2.append(copy.deepcopy(temp))
            self.clfList.append(copy.deepcopy(temp))
        for seg in range(1, numSegs + 1):
            temp = self.overSeg == seg
            self.segL[seg] = np.mean(self.lab[:, :, 0][temp])
            self.segA[seg] = np.mean(self.lab[:, :, 1][temp])
            self.segB[seg] = np.mean(self.lab[:, :, 2][temp])
            for cnnRatioIdx, cnnRes in enumerate(self.allRatioCnnRes):
                for layerNum, outLayer in enumerate(cnnRes):
                    v = outLayer[temp]
                    if self.ifFilter:
                        clf = IsolationForest(behaviour='new', max_samples=max(4, v.shape[0] // self.clfSplit),
                                              n_estimators=self.n_trees,
                                              random_state=0, contamination='auto', n_jobs=1)
                        v_new=v[clf.fit_predict(v)==1,:]
                        if v_new.shape[0]!=0:
                            v = v_new
                    if v.shape[0] > self.maxSize:
                        v = v[np.random.randint(v.shape[0], size=self.maxSize)]
                    self.segVectors[seg][cnnRatioIdx].append(v)
                    cl = np.mean(v, axis=0)
                    self.segClustersL2[seg][cnnRatioIdx].append(cl)

    #Adds a new segment with its information
    def addSeg(self, seg1, seg2, N):
        self.overSeg[np.logical_or(self.overSeg == seg1, self.overSeg == seg2)] = N
        temp = self.overSeg == N
        self.segL[N] = np.mean(self.lab[:, :, 0][temp])
        self.segA[N] = np.mean(self.lab[:, :, 1][temp])
        self.segB[N] = np.mean(self.lab[:, :, 2][temp])

        for cnnRatioIdx, cnnRes in enumerate(self.allRatioCnnRes):
            self.clfList[N][cnnRatioIdx]=[]
            self.segVectors[N][cnnRatioIdx]=[]
            self.segClustersL2[N][cnnRatioIdx]=[]
            for layerNum, outLayer in enumerate(cnnRes):

                v = outLayer[temp]
                if self.ifFilter:

                    clf = IsolationForest(behaviour='new', max_samples=max(4,v.shape[0] // self.clfSplit), n_estimators=self.n_trees,
                                          random_state=0, contamination='auto', n_jobs=1)
                    v_new = v[clf.fit_predict(v) == 1, :]
                    if v_new.shape[0] != 0:
                        v = v_new
                if v.shape[0] > self.maxSize:
                    np.random.seed(0)
                    v = v[np.random.randint(v.shape[0], size=self.maxSize)]
                self.segVectors[N][cnnRatioIdx].append(v)
                cl = np.mean(v, axis=0)
                self.segClustersL2[N][cnnRatioIdx].append(cl)


