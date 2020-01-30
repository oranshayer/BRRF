import scipy.ndimage.morphology as morph
import scipy.ndimage
import scipy.stats
from auxilaryFunc import *
import heapq
from tqdm import tqdm
import skimage.measure as measure



#Removes small segs in the oversegmentation
def removeSmallSegs(EdgeMap, minSegSize=64):
    overSeg = measure.label(EdgeMap == 0)
    overSegSmall= overSeg[1::2,1::2]

    numSegs = overSeg.max()

    segSizes = [None]
    segDilated = [None]
    smallSegHeap = []

    for seg in range(1, numSegs + 1):
        temp = overSegSmall == seg
        segSize = np.count_nonzero(temp)
        segSizes.append(segSize)
        segDilated.append(morph.binary_dilation(overSeg==seg))
        if segSize < minSegSize:
            heapq.heappush(smallSegHeap,(segSize,seg))

    while len(smallSegHeap)>0:
        seg = heapq.heappop(smallSegHeap)[1]
        if segSizes[seg] >= minSegSize:
            continue
        minEdge = np.inf
        temp = overSegSmall == seg
        segNeighbors = np.unique(overSegSmall[morph.binary_dilation(temp)])
        segNeighbors = segNeighbors[segNeighbors != seg]
        for neighbor in segNeighbors:
            segBound = np.logical_and(segDilated[seg],segDilated[neighbor])
            edgeVals = EdgeMap[segBound]
            if np.max(edgeVals) ==0:
                continue
            boundVals = edgeVals[edgeVals>0]
            neighborEdge = np.mean(boundVals)
            if neighborEdge < minEdge:
                mostSimilarNeighbor = neighbor
                minEdge = neighborEdge
                edgeLength=boundVals.size
            elif neighborEdge == minEdge and edgeLength < boundVals.size:
                mostSimilarNeighbor = neighbor
                minEdge = neighborEdge
                edgeLength = boundVals.size
        segSizes[mostSimilarNeighbor] += segSizes[seg]
        segDilated[mostSimilarNeighbor] = np.logical_or(segDilated[seg],segDilated[mostSimilarNeighbor])
        overSeg[overSeg==seg]=mostSimilarNeighbor

    newEdgeMap = EdgeMap.copy()
    temp = scipy.ndimage.convolve(overSeg,np.array([[-1,0,1]]))
    temp2 = scipy.ndimage.convolve(overSeg, np.array([[1, 0, 1]]))
    placeToReplace=np.logical_and(temp==0,temp2>0)[1::2,2::2]
    newEdgeMap[1::2,2::2][placeToReplace]=0
    temp = scipy.ndimage.convolve(overSeg, np.array([[-1], [0], [1]]))
    temp2 = scipy.ndimage.convolve(overSeg, np.array([[1], [0], [1]]))
    placeToReplace=np.logical_and(temp==0,temp2>0)[2::2,1::2]
    newEdgeMap[2::2,1::2][placeToReplace] = 0
    temp =  scipy.ndimage.maximum_filter(newEdgeMap,footprint=np.array([[0,1,0],[1,0,1],[0,1,0]]))
    newEdgeMap[2::2,2::2] = temp[2::2,2::2]
    return newEdgeMap


