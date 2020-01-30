import scipy.ndimage.morphology as morph
import scipy.ndimage
import scipy.stats
from auxilaryFunc import *
import heapq
from tqdm import tqdm
import skimage.measure as measure
from scipy.spatial.distance import pdist, squareform, cdist


#Predicts the dissimilarity between two segments, puts in scoreHeap for fast retrieval
def predict(clf, seg1,seg2, vectorData, segData, scoreHeap):
    myX = getFeaturesFromVectors_pair(vectorData, seg1, seg2, segData)
    x = getFeaturesFromMyX(myX, vectorData.ratios)
    x = np.array(x)
    try:
        pair_dissim = clf.predict_proba(x.reshape(1, -1))[0, 1]
    except:
        pair_dissim = clf.predict(x.reshape(1, -1))[0]
    heapq.heappush(scoreHeap, (pair_dissim, (seg1, seg2)))
    return pair_dissim


#Segmentation without re-ranking
def segmentImage_pair(vectorData, clf, verbose = True):
    np.random.seed(0)
    E_ucm = vectorData.ucm[0]
    overSeg = measure.label(E_ucm==0)
    overSegSmall= overSeg[1::2,1::2]

    numSegs = overSeg.max()

    active = np.zeros(numSegs * 2 )
    active[1: numSegs+1]=1

    resUCM = np.zeros(E_ucm.shape)
    segData = SegData(overSeg=overSeg, ucm=E_ucm)

    neighbors = segData.neighbors
    segDilated = segData.segDilated


    scoreHeap = []

    for seg1 in range(1, overSeg.max() + 1):
        for seg2 in neighbors[seg1]:
            if seg1 <= seg2:
                continue
            predict(clf, seg1,seg2, vectorData, segData, scoreHeap)

    N = numSegs + 1
    remainingSegs = numSegs - 1
    for rep in tqdm(range(numSegs-1), disable=not verbose):

        el = heapq.heappop(scoreHeap)
        combSeg = el[-1]

        while not np.all(active[np.array(combSeg) ] == 1):
            el = heapq.heappop(scoreHeap)
            combSeg = el[-1]

        seg1 = combSeg[0]
        seg2 = combSeg[1]

        overSegSmall[np.logical_or(overSegSmall==seg1, overSegSmall==seg2)] = N

        vectorData.addSeg(seg1, seg2, N)

        segData.addSegComb(seg1,seg2,N)

        if remainingSegs > SP_RANGE:
            resUCM[np.logical_and(segDilated[seg1], segDilated[seg2])] = (rep + 1) / (numSegs - (SP_RANGE - 1)) * MAX_SP_VAL
        else:
            resUCM[np.logical_and(segDilated[seg1], segDilated[seg2])] = (SP_RANGE - remainingSegs) / SP_RANGE * (1 - MAX_SP_VAL) + MAX_SP_VAL

        active[N ] = 1
        active[seg1 ] = 0
        active[seg2 ] = 0

        for seg2 in neighbors[N]:
            predict(clf, N,seg2, vectorData, segData, scoreHeap)
        N += + 1
        remainingSegs -= 1

    holes = np.logical_and(resUCM == 0, E_ucm)
    resUCM[holes] = (scipy.ndimage.maximum_filter(resUCM, size=3))[holes]
    return resUCM



#def mySil(seg, neighbors, vectorData, metric='euclidean'):
#    vectors = vectorData.segVectors
#    A = np.sum(squareform(pdist(vectors[seg][0][0], metric=metric)),axis=1)/(vectors[seg][0][0].shape[0]-1)
#
#    B=np.ones(vectors[seg][0][0].shape[0])*np.inf
#    for seg2 in neighbors:
#        temp = np.sum(cdist(vectors[seg][0][0], vectors[seg2][0][0], metric=metric),axis=1)/(vectors[seg2][0][0].shape[0])
#        B= np.minimum(B,temp)
#
#    res = (B - A) /np.maximum(B,A)
#    return np.mean(res)


#Sil score calculation, is slightly different than the commented one
# (unifying neighboring segments rather than looking at them individually)
def mySil(seg, neighbors, vectorData, metric='euclidean'):
    if len(neighbors)==0:
        return -1
    vectors = vectorData.segVectors
    A = np.sum(squareform(pdist(vectors[seg][0][0], metric=metric)),axis=1)/(vectors[seg][0][0].shape[0]-1)

    neighbors_samples = None
    for seg2 in neighbors:
        if neighbors_samples is None:
            neighbors_samples=vectors[seg2][0][0]
        else:
            neighbors_samples=np.vstack((neighbors_samples,vectors[seg2][0][0]))
    B = np.sum(cdist(vectors[seg][0][0], neighbors_samples, metric=metric),axis=1)/(neighbors_samples.shape[0])

    res = (B - A) /np.maximum(B,A)
    return np.mean(res)


#Segmentation with re-ranking, it's code duplication but it's safer to edit
def segmentImage_pair_rerank(vectorData, clf, verbose = True, startFromSeg = 120, numSegsToTest = 4):
    np.random.seed(0)
    E_ucm = vectorData.EdgeMapList[0]
    overSeg = measure.label(E_ucm==0)
    overSegSmall= overSeg[1::2,1::2]

    numSegs = overSeg.max()

    active = np.zeros(numSegs * 2 )
    active[1: numSegs+1]=1

    resUCM = np.zeros(E_ucm.shape)
    segData = SegData(overSeg=overSeg, ucm=E_ucm)

    neighbors = segData.neighbors
    segDilated = segData.segDilated

    scoreHeap = []

    for seg1 in range(1, overSeg.max() + 1):
        for seg2 in neighbors[seg1]:
            if seg1 <= seg2:
                continue
            predict(clf, seg1,seg2, vectorData, segData, scoreHeap)

    N = numSegs + 1
    remainingSegs = numSegs - 1

    testedDict=dict()

    valuesAr=[]

    for rep in tqdm(range(numSegs-1), disable=not verbose):

        el = heapq.heappop(scoreHeap)
        combSeg = el[-1]

        while not np.all(active[np.array(combSeg) ] == 1):
            el = heapq.heappop(scoreHeap)
            combSeg = el[-1]

        if remainingSegs > 1 and remainingSegs < startFromSeg:
            segsToTest = [el]

            for _ in range(numSegsToTest-1):
                try:
                    el = heapq.heappop(scoreHeap)
                    combSeg = el[-1]
                    while not np.all(active[np.array(combSeg)] == 1):
                        el = heapq.heappop(scoreHeap)
                        combSeg = el[-1]
                    segsToTest.append(el)
                except:
                    pass

            segsScores = []
            for el in segsToTest:
                combSeg = el[-1]
                seg1 = combSeg[0]
                seg2 = combSeg[1]
                neighborsN = neighbors[seg1] | neighbors[seg2]
                neighborsN.difference_update([seg1, seg2])
                combId= combSeg
                combId2 = tuple(neighborsN)
                combId = combId +combId2
                if combId in testedDict:
                    score = testedDict[combId]
                else:
                    vectorDataOverSegCopy = vectorData.overSeg.copy()
                    vectorData.addSeg(seg1, seg2, N)
                    segData.addSegComb(seg1, seg2, N, recalcNeighbors=False)
                    score = (1 - el[0]) + 0.5 * mySil(N, neighbors[N], vectorData, metric='euclidean')
                    testedDict[combId] = score
                    vectorData.overSeg = vectorDataOverSegCopy
                segsScores.append(score)

            for el in segsToTest:
                heapq.heappush(scoreHeap,el)

            maxIdx = np.argmax(segsScores)
            el = segsToTest[maxIdx]
            combSeg = el[-1]

        valuesAr.append(el[0])

        seg1 = combSeg[0]
        seg2 = combSeg[1]

        overSegSmall[np.logical_or(overSegSmall==seg1, overSegSmall==seg2)] = N

        vectorData.addSeg(seg1, seg2, N)

        segData.addSegComb(seg1,seg2,N)


        resUCM[np.logical_and(segDilated[seg1], segDilated[seg2])] = rep + 1

        active[N ] = 1
        active[seg1 ] = 0
        active[seg2 ] = 0

        for seg2 in neighbors[N]:
            predict(clf, N,seg2, vectorData, segData, scoreHeap)

        N += + 1
        remainingSegs -= 1

    holes = np.logical_and(resUCM == 0, E_ucm)
    resUCM[holes] = (scipy.ndimage.maximum_filter(resUCM, size=3))[holes]
    try:
        valuesAr_new = scipy.signal.savgol_filter(np.array(valuesAr + [1]),11,1)
    except:
        valuesAr_new = np.array(valuesAr + [1])
    valuesAr_new = np.round(valuesAr_new,3)
    val = valuesAr_new[0]
    valIdx = 0
    for i in range(1, len(valuesAr_new)):
        if valuesAr_new[i] > val:
            if i - valIdx > 1:
                valuesAr_new[valIdx + 1:i] = np.linspace(valuesAr_new[valIdx], valuesAr_new[i], i - valIdx + 1)[1:-1]
            val = valuesAr_new[i]
            valIdx = i
    valuesAr_new=valuesAr_new[:-1]
    for idx,val in enumerate(valuesAr_new):
        resUCM[resUCM==(idx+1)]=val
    return resUCM
