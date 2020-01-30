from scipy.io import savemat
import numpy as np
import scipy.stats
import matlab.engine
from collections import Counter, namedtuple
import scipy.spatial.distance
import scipy.signal
import matplotlib.pyplot as plt
import bottleneck as bn
import os
import cv2
from SegData import SegData
import scipy.ndimage.morphology as morph
from sklearn.metrics.pairwise import cosine_distances
import skimage.measure as measure


databaseDir=os.sep.join(['.', 'BSR', 'BSDS500', 'data'])
cobResDir = 'COB_res'
XYRepsFolder = 'XY'
repDirUsed = 'RepNet_res'
edgeThresList = [0.08, 0.1, 0.15, 0.2, 0.25, 0.3]

np.seterr(all='raise')
eng = None

#Class that defines features between two segments
class myX:
    def __init__(self):
        self.regFeatures = None #boundary and shape/size features
        self.cnnFeatures = dict() #boundary relation to region representation
        self.ImageFeatures = None #LAB color features


# Return features from myX
def getFeaturesFromMyX(myX,ratios):
    features = myX.regFeatures.copy()
    features[2:4] = list(np.sqrt(np.array(features[2:4])))
    for ratio in ratios:
        for idx, cnnLayer in enumerate(myX.cnnFeatures[ratio]):
            cnnLayer[:5] = np.log( 1 + np.array(cnnLayer[:5]))
            features.extend(cnnLayer)
    features.extend(myX.ImageFeatures)
    return features


#Returnn a score based on Seism implementation in Matlab
def getSegEval(resName, newUCM,oldUCM,GTName,ImageName, calcFop, deleteAftwerwards=False):
    savemat(resName,dict(newUCM=newUCM, oldUCM=oldUCM, GTName=GTName, ImageName=ImageName))
    global eng
    if eng is None:
        eng = matlab.engine.start_matlab()
    try:
        if calcFop:
            res = eng.testSegRes(resName)
        else:
            print("Calc Fb")
            res = eng.testSegResFb(resName)
    except:
        raise ("Evaluation of segmentation fail. Did you set seism-master dir in matlab files?")
    if deleteAftwerwards:
        os.remove(resName + ".mat")
    return res['p_best'], res['r_best']


#Returns myX object for a pair of segments
def getFeaturesFromVectors_pair(vectorData, seg1, seg2, segData ):
    X = myX()
    segDilated=segData.segDilated
    segBoundSize = segData.segBoundSize
    seggSizes = segData.segSizes
    EdgeMapList = vectorData.EdgeMapList
    vectors = vectorData.segVectors
    clustersL2 = vectorData.segClustersL2
    ratios = vectorData.ratios
    regFeatures = []
    boundLine = np.logical_and(segDilated[seg1], segDilated[seg2])
    for e in EdgeMapList:
        bound = e[boundLine]
        bound = bound[bound > 0]
        regFeatures.append(np.mean(bound))  #Feature mean bound dist.

    boundOverlap1 = bound.size / segBoundSize[seg1]
    boundOverlap2 = bound.size / segBoundSize[seg2]
    regFeatures.append(max(boundOverlap1, boundOverlap2)) #Feature max overlap with bound
    regFeatures.append(bound.size) #Feature bound size
    size1 = seggSizes[seg1]
    size2 = seggSizes[seg2]
    regFeatures.append(size1 + size2) #Feature New seg area

    X.regFeatures = regFeatures
    for ratioIdx, ratio in enumerate(ratios):
        cnnFeatures = []
        for numLayer in range(0,len(vectors[seg1][ratioIdx])):
            layerFeatures=[]
            pair_dist = scipy.spatial.distance.cdist(vectors[seg1][ratioIdx][numLayer], vectors[seg2][ratioIdx][numLayer])
            layerFeatures.append(np.min(pair_dist)) #Feature L2 min dist in vec rep.
            layerFeatures.append(np.max(pair_dist)) #Feature L2 max dist in vec rep.
            layerFeatures.append(np.mean(pair_dist)) #Feature L2 average dist in vec rep.
            layerFeatures.append(bn.median(pair_dist)) #Feature L2 median dist in vec rep.
            layerFeatures.append(np.sqrt(np.sum((clustersL2[seg1][ratioIdx][numLayer] - clustersL2[seg2][ratioIdx][numLayer]) ** 2))) #Feature L2 dist between L2 clusters
            pair_dist = scipy.spatial.distance.cdist(vectors[seg1][ratioIdx][numLayer],
                                                     vectors[seg2][ratioIdx][numLayer], metric='cosine')
            layerFeatures.append(np.min(pair_dist))  # Feature cosine min dist in vec rep.
            layerFeatures.append(np.max(pair_dist))  # Feature cosine max dist in vec rep.
            layerFeatures.append(np.mean(pair_dist))  # Feature cosine average dist in vec rep.
            layerFeatures.append(bn.median(pair_dist))  # Feature cosine median dist in vec rep.
            layerFeatures.append(cosine_distances(np.array([clustersL2[seg1][ratioIdx][numLayer]]),
                                                  np.array([clustersL2[seg2][ratioIdx][numLayer]]))[0][0]) #Feature cosine dist between L2 clusters

            cnnFeatures.append(layerFeatures)
        X.cnnFeatures[ratio] = cnnFeatures
    ImageFeatures = []
    ImageFeatures.append(np.sqrt((vectorData.segL[seg1] - vectorData.segL[seg2]) ** 2)) #Feature L channel dist
    ImageFeatures.append(np.sqrt((vectorData.segA[seg1] - vectorData.segA[seg2]) ** 2)) #Feature A channel dist
    ImageFeatures.append(np.sqrt((vectorData.segB[seg1] - vectorData.segB[seg2]) ** 2)) #Feature B channel dist
    X.ImageFeatures = ImageFeatures
    return X


#Returns classifier training examples to save
def GetVectorXY_pair(GT_seg, GT_bound, vectorData, overSeg):
    ucm = vectorData.EdgeMapList[0]
    GT_bound_dilated = []
    overSegSmall = overSeg[1::2, 1::2]

    segData = SegData(overSeg=overSeg, ucm=ucm)
    neighbors = segData.neighbors

    segDilated = [None]
    for seg in range(1,overSegSmall.max()+1):
        segDilated.append(morph.binary_dilation(overSeg==seg, iterations=1))

    segGT = []
    for seg in range(1, overSegSmall.max() + 1):
        segTotPrec = []
        for GT in GT_seg:
            temp=Counter(GT[overSegSmall == seg])
            segTotPrec.append(temp.most_common(1)[0][0])
        segGT.append(np.array(segTotPrec))

    for GT in GT_bound:
        temp = cv2.resize(morph.binary_dilation(GT, iterations=2).astype(int), ucm.shape[::-1], interpolation=cv2.INTER_NEAREST)
        GT_bound_dilated.append( temp>0 )

    X = []
    Yb = []

    for seg1 in range(1,overSegSmall.max()+1):
        for seg2 in neighbors[seg1]:
            if seg1<=seg2:
                continue
            bound=np.logical_and(segDilated[seg1],segDilated[seg2])
            boundLen=np.count_nonzero(bound)
            GT_cross = [np.count_nonzero(np.logical_and(bound,GT))/boundLen for GT in GT_bound_dilated]
            Yb.append(GT_cross)
            X.append(getFeaturesFromVectors_pair(vectorData,seg1,seg2, segData))
    return (X, Yb)
