function shapeModel = BuildShapeModel( trainImageDataList, numFacePoints )
    numTrainData = length( trainImageDataList );
    
    % PCA
    matTrainFace = zeros( numTrainData, 2*numFacePoints )
    for i = 1:length(trainImageDataList)
       trainData = trainImageDataList(i);
       % get vector
       x = trainData.aligned_x;
       y = trainData.aligned_y;

       vect = GetXYPointsVector(x, y);

       matTrainFace(i,:) = vect;
    end
    % take transpose so each face is represented by column
    matTrainFace = transpose( matTrainFace );

    % eigen decomposition, eigenVect is normalized
    [ eigenVect, eigenVal, meanShape ] = PCA( matTrainFace );

    % get the best eigen vectors which accounts to most variations (99%)
    % finds the index where the eigen val sum is just another 99% of total
    cutOffVariance = 0.99;
    cutOffIndx = find( cumsum( eigenVal ) < sum( eigenVal )*cutOffVariance, 1, 'last' ) + 1;

    eigenVal = eigenVal(1:cutOffIndx, 1);
    eigenVect = eigenVect( :, 1:cutOffIndx );

    shapeModel.numEigenVals = length(eigenVal);
    shapeModel.numPts = numFacePoints;
    shapeModel.eigenVals = eigenVal;
    shapeModel.eigenVects = eigenVect;
    shapeModel.meanShape = meanShape;


end

