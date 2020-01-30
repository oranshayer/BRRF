function [res] = testSegRes(resName)
%addpath(genpath('../seism-master'));
load(resName);
Image=imread([ImageName '.jpg']);

% newUCM=1/1+exp(oldUCM);
GT=load(GTName);

GTAllSegs={};
for ii=1:numel(GT.groundTruth)
    GTAllSegs{ii}=GT.groundTruth{ii}.Segmentation;
end
% GTAllSegs = GTSeg;

edgeNumsNew=unique(sort(newUCM(:)));
edgeNumsNew(edgeNumsNew<median(edgeNumsNew))=[];
numSegsNew=zeros(numel(edgeNumsNew),1);
for i=1:numel(edgeNumsNew)
    newSeg=bwlabel(newUCM<=edgeNumsNew(i),4);
    numSegsNew(i)=max(newSeg(:));
end

newF=[];
newR=[];
newP=[];

numSegsList=150:-5:30;
for numSegs=numSegsList
%for numSegs=[12:-1:1]
    thresNew=edgeNumsNew(find(numSegsNew<=numSegs,1));
    newSeg=bwlabel(newUCM<=thresNew,4);
    newSeg=newSeg(2:2:end,2:2:end);

    measure=eval_segm( newSeg, GTAllSegs, 'fb' );
    newF=[newF measure(1)];
    newP=[newP measure(2)];
    newR=[newR measure(3)];
end
[~,idx]= max(newF);
res.p_best = newP(idx);
res.r_best = newR(idx);
end

