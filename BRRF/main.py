from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from auxilaryFunc import *
from SegmentImage import *
from trainModel import *
from VectorData import VectorDataOrig
import skimage.measure as measure
import time
from RemoveSmallSegs import removeSmallSegs
from skimage import io
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=bool, help="Whether to output progress and score calculation (if used)", default=True)
parser.add_argument("--calcScore", type=bool, help="Calculate segmentation evaluation (highest F score). Needs matlab and seism-master installed", default=False)
parser.add_argument("--calcFop", type=bool, help="Do the calculation using Fop, the alternative is Fob.", default=True)
parser.add_argument("--del_eval_afterwards", type=bool, help="Deletes segmentation evaluation files afterwards", default=True)
parser.add_argument("--rerank", type=bool, help="Use Reranking", default=True)
parser.add_argument("--generateXY", type=bool, help="Generate XY training examples for classifier", default=False)
parser.add_argument("--overwriteXY", type=bool, help="Overwrite XY training examples in generation", default=False)
parser.add_argument("--classType", type=str, help="Classifier type for training. Options are 'lr', 'mlp', 'svm, 'for'", default='lr')
parser.add_argument("--createUCMDir", type=bool, help="Saves resulting ucm in directory in a format for seism-master", default=True)
parser.add_argument("--newUCMDir", type=str, help="Result ucm directory", default='BRRF_Res')
parser.add_argument("--trainOnly", type=bool, help="Use only train folder in training and evaluates on val set (Need bsds format)", default=True)
args = parser.parse_args()


calcScore = args.calcScore
calcFop = args.calcFop
rerank = args.rerank
generateXY = args.generateXY
overwriteXY = args.overwriteXY
verbose = args.verbose
createUCMDir = args.createUCMDir
classType = args.classType
newUCMDir = args.newUCMDir
del_eval_afterwards = args.del_eval_afterwards
trainOnly = args.trainOnly

#calcScore = False
#calcFop = True
#rerank = True
#generateXY = False
#overwriteXY = True
#verbose = True
#createUCMDir = False
#classType = 'lr'
#newUCMDir = 'BRRF_Res'
#del_eval_afterwards = True
#trainOnly = True

startRerankFromSeg = 120
numSegsToTestRerank = 4
components = 9
ifFilter = True
sampleSize = 300

if trainOnly:
    #print ("Checking val")
    evalTestDir= 'val'
    numImages = 100
else:
    #print ("Checking test")
    evalTestDir = 'test'
    numImages = 200


#Tests single image
def testSingleImage(imageNum, ratios, clfName, components=9, verbose=True, minSegSize=32, sampleSize=300, ifFilter=True):
    imageDir = os.sep.join([databaseDir, 'images', evalTestDir])

    imagesNames = os.listdir(imageDir)
    imagesNames = list(filter(lambda name: name.endswith('.jpg'), imagesNames))
    imagesNames.sort()

    clf = pickle.load(open(clfName, "rb"))
    name = imagesNames[imageNum]

    if (verbose):
        print("Image No. " + str(imageNum))
    Image = io.imread(imageDir + os.sep + name)
    name = name[:-4]


    ucmMat = loadmat(os.sep.join([cobResDir, name + '.mat']))
    E1 = ucmMat['E1']
    allRatioCnnRes = []

    cnnResName = os.sep.join([repDirUsed + os.sep + name + '.npy'])
    layer_outs = [np.load(open(cnnResName, "rb"))]
    for idx, out in enumerate(layer_outs):
        res_im = np.squeeze(out)
        res_vec = res_im.reshape((res_im.shape[0] * res_im.shape[1], res_im.shape[2]))
        res_reduced = PCA(n_components=components).fit_transform(res_vec)
        res_reduced = np.reshape(res_reduced, (res_im.shape[0], res_im.shape[1], components))
        cnnRes = [res_reduced]
    allRatioCnnRes.append(cnnRes)
    np.random.seed(0)
    E1 = removeSmallSegs(E1, minSegSize=minSegSize)
    overSegNew = measure.label(E1 == 0)
    vectorData = VectorDataOrig(EdgeMapList=[E1], overSeg=overSegNew, ratios=ratios, allRatioCnnRes=allRatioCnnRes,
                                maxSize=sampleSize, Image=Image, ifFilter=ifFilter)
    if rerank:
        resUCM = segmentImage_pair_rerank(vectorData, clf, verbose=verbose, startFromSeg=startRerankFromSeg, numSegsToTest=numSegsToTestRerank)
    else:
        resUCM = segmentImage_pair(vectorData, clf, verbose=verbose)
    if calcScore:
        resName = str(imageNum)
        gtDir = os.sep.join([databaseDir, 'groundTruth', evalTestDir])
        GTName = gtDir + os.sep + name + '.mat'
        p_best,r_best = getSegEval("ucm_comp_dir" + os.sep + resName, resUCM, resUCM, GTName, imageDir + os.sep + name, calcFop, deleteAftwerwards=del_eval_afterwards)
        if verbose:
            print("Res: " + str(2*p_best*r_best/(p_best+r_best)))
    else:
        p_best = 0
        r_best = 0
    if createUCMDir:
        savemat(os.sep.join([newUCMDir, name]), dict(ucm2=resUCM))
    return p_best, r_best



if __name__ == '__main__':
    ratios = [1]
    imagesToTest = range(0, numImages)
    if generateXY:
        import generateExamples
        #Increased minSegSize is due to too many small segments pairs causing a bias in generation
        generateExamples.generateExamples(overWrite=overwriteXY, components=components, minSegSize=64, output_folder_name=XYRepsFolder,
                                          trainOnly=trainOnly, repUsed=repDirUsed, verbose=verbose)
    clfName = "trained_clf" + os.sep + "clf"
    train_clf(edgeThresList=edgeThresList, ratios=ratios,
                        clfName=clfName, classType=classType,
                        verbose=False, trainOnly=trainOnly)
    res_p = []
    res_r = []
    t = time.time()
    for imageNum in imagesToTest:
        p_best, r_best = testSingleImage(imageNum=imageNum, ratios=ratios, components=components,
                                  clfName=clfName, verbose=verbose, minSegSize = 32, ifFilter=ifFilter, sampleSize=sampleSize)
        res_p.append(p_best)
        res_r.append(r_best)
    mean_best_p = np.mean(res_p)
    mean_best_r = np.mean(res_r)
    res =  (2*mean_best_p*mean_best_r) / (mean_best_p + mean_best_r + 1e-6)
    timeTook = (time.time() - t) / 60
    if calcScore:
        print('Mean P is' + " : " + str(mean_best_p) + '\n')
        print('Mean R is' + " : " + str(mean_best_r) + '\n')
        print('Result is' + " : " + str(res) + '\n')
    print("Time took (Min): " + str(timeTook))
    print("Avg. time for image (Min): " + str(timeTook/numImages))

