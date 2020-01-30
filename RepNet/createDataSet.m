close all
clearvars
clc
dbstop if error

%% Create segmentations matrices

dirname_seg = 'trainval_orig_seg';
dirname_GT = 'trainval_GT';
list=dir(dirname_GT); list = list(3:end);


if isfolder(dirname_seg)==0
    mkdir(dirname_seg);
end
for i=1:length(list)
    load([dirname_GT filesep list(i).name]);
    imSeg = uint8(zeros([size(groundTruth{1}.Segmentation) length(groundTruth)]));
    for j=1:length(groundTruth)
        imSeg(:,:,j)=groundTruth{j}.Segmentation;
    end
    save([dirname_seg '/' list(i).name], 'imSeg');
end
%% Create edge maps and segs

dirname_edges = 'trainval_edges';
dirname_labels = 'trainval_labels';
dirname_seg = 'trainval_orig_seg';
list=dir(dirname_seg); list = list(3:end);

if isfolder(dirname_edges)==0
    mkdir(dirname_edges);
end
if isfolder(dirname_labels)==0
    mkdir(dirname_labels);
end
NUM_CLASSES = 0;
percent_edges = 0;
for i=1:length(list)
    load([dirname_seg filesep list(i).name]);
    origSeg = imSeg;
    s = size(imSeg);
    numSegments = zeros(1,s(3));
    for j=1:s(3)
        numSegments(j) = length(unique(imSeg(:,:,j)));
    end
    [Y,I] = sort(numSegments);
    idx = I(ceil(length(numSegments)/2)); %get the median seg length

    imSeg = imSeg(:,:,idx);
    edge = boundarymask(imSeg,8);
    elements = unique(imSeg);
    for j=1:length(elements)
        numPixels = sum(sum(imSeg==elements(j)));
        if numPixels<2500, imSeg(imSeg==elements(j)) = 0; 
        end
    end
    elements = unique(imSeg); if elements(1)==0, elements = elements(2:end); end
    newIndices = (NUM_CLASSES+1):(NUM_CLASSES+length(elements));
    newSeg = zeros(s(1),s(2),'int32');
    for j=1:length(elements)
        newSeg(imSeg==elements(j)) = newIndices(j);
    end
    imSeg = newSeg;
    NUM_CLASSES = NUM_CLASSES+length(elements);
    
    edge = imdilate(edge,strel('disk',4));
    
    
    new_filename = strrep(list(i).name,'mat','bin');
    fileID = fopen([dirname_edges filesep new_filename],'w');
    fwrite(fileID,edge);
    fclose(fileID);
    
    fileID = fopen([dirname_labels filesep new_filename],'w');
    fwrite(fileID,imSeg,'int32');
    fclose(fileID);
end

%% Create label maps
dirname_poss = 'trainval_poss_lbls_';
if isfolder(dirname_poss)==0
    mkdir(dirname_poss);
end
list=dir(dirname_labels); list = list(3:end);

for i=1:length(list)
    new_filename = strrep(list(i).name,'mat','bin');
    fileID = fopen([dirname_labels filesep new_filename],'r');
    imSeg = fread(fileID,'int32');
    fclose(fileID);
    poss_lbls = ones(35,35,NUM_CLASSES+1,'uint8'); 
    uniq = unique(imSeg);
    poss_lbls(:,:,uniq+1) = 0;
    poss_lbls = permute(poss_lbls,[3,2,1]); % need to permute because TF reshape works the opposite
    fileID = fopen([dirname_poss filesep new_filename],'w');
    fwrite(fileID,poss_lbls);
    fclose(fileID);
end

%% Create summary

fileID = fopen('summary.txt','w');
fprintf(fileID, strcat('Num classes = ',' ', string(NUM_CLASSES+1)));
fclose(fileID);
disp(NUM_CLASSES+1);

