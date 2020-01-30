import numpy as np
import scipy.ndimage.morphology as morph

#Class designed to keep geometric information about the segments
class SegData:
    def __init__(self, ucm, overSeg):
        numSegs = overSeg.max()
        neighbors = [None]* (numSegs * 2)
        segDilatedSmall = [None] * (numSegs * 2)
        segDilated = [None]* (numSegs * 2)
        segSizes = [None]* (numSegs * 2)
        segBound = [None]* (numSegs * 2)
        segBoundSize = [None]* (numSegs * 2)
        overSegSmall = overSeg[1::2, 1::2]
        ucmTemp = ucm.copy()
        ucmTemp[0,:]=1
        ucmTemp[-1,:]=1
        ucmTemp[:,0] = 1
        ucmTemp[:,-1] = 1
        for seg in range(1, overSeg.max() + 1):
            temp = overSegSmall == seg
            segDilatedSmall[seg] = morph.binary_dilation(temp)
            segNeighbors = np.unique(overSegSmall[segDilatedSmall[seg]])
            segSizes[seg] = np.count_nonzero(temp)
            segNeighbors = segNeighbors[segNeighbors != seg]
            neighbors[seg]=set(segNeighbors)
            temp2 = overSeg == seg
            segDilated[seg]=morph.binary_dilation(temp2)
            segBound[seg]=np.logical_and(segDilated[seg],ucmTemp)
            segBoundSize[seg] = np.count_nonzero(segBound[seg])
        self.segDilatedSmall = segDilatedSmall
        self.segDilated=segDilated
        self.neighbors=neighbors
        self.segSizes=segSizes
        self.segBound=segBound
        self.segBoundSize=segBoundSize
        self.ucm=ucm


    #Adds a new segment
    def addSegComb(self,seg1, seg2, N, recalcNeighbors = True):
        neighborsN = self.neighbors[seg1] | self.neighbors[seg2]
        neighborsN.difference_update([seg1, seg2])
        self.neighbors[N]=neighborsN
        if recalcNeighbors:
            for seg in range(1, N):
                if seg1 in self.neighbors[seg]:
                    self.neighbors[seg].remove(seg1)
                    self.neighbors[seg].add(N)
                if seg2 in self.neighbors[seg]:
                    self.neighbors[seg].remove(seg2)
                    self.neighbors[seg].add(N)
        self.segDilatedSmall[N]=np.logical_or(self.segDilatedSmall[seg1], self.segDilatedSmall[seg2])
        self.segDilated[N] = np.logical_or(self.segDilated[seg1], self.segDilated[seg2])
        self.segSizes[N]=self.segSizes[seg1] + self.segSizes[seg2]
        self.segBound[N]=np.logical_xor(self.segBound[seg1], self.segBound[seg2])
        self.segBoundSize[N]=np.count_nonzero(self.segBound[N])




