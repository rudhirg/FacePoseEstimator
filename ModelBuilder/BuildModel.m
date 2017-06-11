% Step 1. Read Annotated files
% Step 2. procustes ananlysis
% Step 3. PCA

% params
params.CanvasSize = [400 400];
params.PatchSize = [32 32];
% SVM C parameter
params.SvmC=5e-3;
params.debugPatches = 0;

% folder containing the annotated data for images
annotatedFileDir = '../data/points/';
% folder containing the training images
trainImgFolder = '../data/images/';

% get the name of all files in images folder
imageFileList = dir( trainImgFolder );
% get the name of all files in points folder
pointFileList = dir( annotatedFileDir );

imageFileDict = containers.Map;
pointFileDict = containers.Map;

% number of training images
numTrainData = 0;

% number of face points
numFacePoints = 68;

%% Data Parsing

% for each fileName, get the annotated points
for i = 1:length(pointFileList)
    fileName = pointFileList(i);
    fileName = fileName.name;
    if strcmp(fileName, '.') || strcmp(fileName, '..')
        continue;
    end
    % get rid of extension
    filePrefix = fileName(1:findstr(fileName, '.')-1);
    pointFileDict( filePrefix ) = fileName;
end
    
% for each fileName, get the image
for i = 1:length(imageFileList)
    fileName = imageFileList(i);
    fileName = fileName.name;
    if strcmp(fileName, '.') || strcmp(fileName, '..')
        continue;
    end
    % get rid of extension
    filePrefix = fileName(1:findstr(fileName, '.')-1);
    imageFileDict( filePrefix ) = fileName;
end

% global variable that stores all the image details
trainImageDataList = [];

% Read all the annotated points and store it
for val = imageFileDict.keys()
    trainImageData = TrainImageData;
    trainImageData.filePrefix = val;
    trainImageData.fileImageName = imageFileDict(char(val));
    trainImageData.filePointName = pointFileDict(char(val));
    % read image
    img = imread( [trainImgFolder '/' trainImageData.fileImageName] );
    img = im2double( rgb2gray( img ) );
    trainImageData.img = img;
    
    % get annotated points
    [x, y] = ReadImagePoints(trainImageData.filePointName, annotatedFileDir);
    trainImageData.orig_x = x;
    trainImageData.orig_y = y;
    
    trainImageDataList = [trainImageDataList trainImageData];
end

numTrainData = length( trainImageDataList );

%% Preprocessing
% preprocess first face
[newxy, img] = PreprocessFirstFace( trainImageDataList(1), params );
trainImageDataList(1).img = img;
trainImageDataList(1).aligned_x = newxy(:,1)';
trainImageDataList(1).aligned_y = newxy(:,2)';
% preprocess all other training faces according to first face
for i = 2:length(trainImageDataList)
   trainImageData = trainImageDataList(i);
   [ newXy, newImg ] = PreprocessTrainFace( trainImageDataList(1), trainImageData, params );
   trainImageDataList(i).img = newImg;
   trainImageDataList(i).aligned_x = newXy(:, 1)';
   trainImageDataList(i).aligned_y = newXy(:, 2)';
end


% % procrustes analysis
% % make X matrix from the first image
% x = trainImageDataList(1).points_x;
% y = trainImageDataList(1).points_y;
% X = GetFacePointsMatrix( x, y );
% 
% for i = 1:length(trainImageDataList)
%    trainData = trainImageDataList(i);
%    % make Y matrix
%    x = trainData.points_x;
%    y = trainData.points_y;
%    Y = GetFacePointsMatrix( x, y );
%    
%    % procrustes
%    [d, Z, tr ] = ProcustesAnalysis( X, Y );
%    
%    trainData.aligned_x = transpose(Z(:, 1));
%    trainData.aligned_y = transpose(Z(:, 2));
%    
%    trainData.align_Tr = tr;
% end

%% build shape model
shapeModel = BuildShapeModel( trainImageDataList, numFacePoints );

% Show some eigenvector variations
if(0)
    figure;
    for i=1:6
        xtest = shapeModel.meanShape + shapeModel.eigenVects(:,i)*sqrt(shapeModel.eigenVals(i))*3;
        subplot(2,3,i), hold on;
        DrawFaceShape(xtest(1:2:end),xtest(2:2:end),'r');
        DrawFaceShape(shapeModel.meanShape(1:2:end),shapeModel.meanShape(2:2:end),'b');
    end
end

%% build patch model
patchModel = BuildPatchModel( trainImageDataList, params );

Model = struct;
Model.shapeModel = shapeModel;
Model.patchModel = patchModel;

% Save XML file.
xml_str = xml_formatany(Model);
fd = fopen('TrainModel.xml', 'w');
if(fd > 0)
    fprintf(fd, '<?xml version="1.0"?>\n');
    fprintf(fd, '%s', xml_str);
    fclose(fd);
else
    fprintf(0, 'Cannot open xml file for writing!');
end

clear