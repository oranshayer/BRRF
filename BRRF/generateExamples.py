from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from auxilaryFunc import *
from VectorData import VectorDataOrig
import skimage.measure as measure
from RemoveSmallSegs import removeSmallSegs
from skimage import io


#Generates training examples for the classifier. Results are saved in 'output_folder_name'
def generateExamples(overWrite=False, components = 9, minSegSize = 64, output_folder_name="XY", trainOnly = True, repUsed=None, verbose=True):

    #TODO: change if there is no division into folders like in BSDS500
    if trainOnly:
        trainEvalDirs=['train']
    else:
        trainEvalDirs = ['train', 'val']

    for trainEvalTestDir in trainEvalDirs:
        imageDir=os.sep.join([databaseDir, 'images', trainEvalTestDir])
        gtDir = os.sep.join([databaseDir, 'groundTruth', trainEvalTestDir])
        imagesNames=os.listdir(imageDir)
        imagesNames = list(filter(lambda name: name.endswith('.jpg'), imagesNames))
        i=0
        ratios = [1]
        for name in imagesNames:
            i += 1
            if verbose:
                print(trainEvalTestDir + " " + str(i) + '-' +  str(name[:-4]))
            Image=io.imread(imageDir + os.sep + name)
            name=name[:-4]
            ucmMat = loadmat(os.sep.join([cobResDir, name + '.mat']))
            ucm_orig = ucmMat['ucm']
            E1 = ucmMat['E1']
            GTName = gtDir + os.sep + name + '.mat'
            GTMat = loadmat(GTName)
            numGT = len(GTMat['groundTruth'][0])
            GTSeg = []
            GTBound = []
            for idx in range(numGT):
                GTSeg.append(GTMat['groundTruth'][0][idx]['Segmentation'][0][0])
                GTBound.append( GTMat['groundTruth'][0][idx]['Boundaries'][0][0])
            allRatioCnnRes = []
            cnnResName = os.sep.join([repUsed + os.sep + name + '.npy'])
            layer_outs = [np.load(open(cnnResName, "rb"))]

            for idx, out in enumerate(layer_outs):
                res_im = np.squeeze(out)
                res_vec = res_im.reshape((res_im.shape[0] * res_im.shape[1], res_im.shape[2]))
                res_reduced = PCA(n_components=components).fit_transform(res_vec)
                res_reduced = np.reshape(res_reduced, (res_im.shape[0], res_im.shape[1], components))
                cnnRes = [res_reduced]
            allRatioCnnRes.append(cnnRes)

            for edgeThres in edgeThresList:
                xyName = os.sep.join(
                    [output_folder_name, name + "_XY_" + str(edgeThres) + ".p"])
                if os.path.exists(xyName) and not overWrite:
                    continue
                else:
                    ucm_orig = np.multiply(ucm_orig, ucm_orig > edgeThres)
                    ucm_orig = removeSmallSegs(ucm_orig, minSegSize=minSegSize)
                    E1 = np.multiply(E1, ucm_orig > 0)
                    overSeg = measure.label(E1 == 0)
                    np.random.seed(0)
                    #VectorData slightly different than in main.py due to optimizations of val sets
                    vectorData = VectorDataOrig(EdgeMapList=[E1], overSeg=overSeg, ratios=ratios, allRatioCnnRes=allRatioCnnRes, maxSize=450, clfSplit=2, Image=Image)
                    (X, Yb)=GetVectorXY_pair(GTSeg,GTBound, vectorData, overSeg)
                    pickle.dump((X, Yb), open(xyName, "wb"))
    if verbose:
        print("Finished generation")


if __name__ == '__main__':
    generateExamples(overWrite=True, components=9, minSegSize=64, output_folder_name=XYRepsFolder, trainOnly=False, repUsed=repDirUsed)

