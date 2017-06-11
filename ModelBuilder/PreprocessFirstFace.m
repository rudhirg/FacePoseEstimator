function [ newxy, img ] = PreprocessFirstFace( trainImageData, Params )
    % crop the first image and scale it to a specific size
    x = trainImageData.orig_x';
    y = trainImageData.orig_y';
    
    minxy = min([x y]);
    maxxy = max([x y]);
    
    width = maxxy(1) - minxy(1);
    height = maxxy(2) - minxy(2);
    
    marginW = width/6;
    marginH = height/6;
    margin = [marginW marginH];
    
    cropMin = minxy - margin;
    cropMax = maxxy + margin;
    
    offset = [0 0];
    if cropMin(1) <= 0
       cropMin(1) = 1;
       offset(1) = -cropMin(1);
    end
    if cropMin(2) <= 0
       cropMin(2) = 1;
       offset(2) = -cropMin(2);
    end
    
    if cropMax(1) >= size( trainImageData.img, 2 )
       cropMax(1) = size( trainImageData.img, 2 ) - 1;
    end
    if cropMax(2) >= size( trainImageData.img, 1 )
       cropMax(2) = size( trainImageData.img, 1 ) - 1;
    end
    
    % scale factor
    newWidHt = cropMax - cropMin;
    newWid = newWidHt(1);
    newHt = newWidHt(2);
    
    scaleF = Params.CanvasSize(1)/width;
    if scaleF*height > Params.CanvasSize(2)
       scaleF = Params.CanvasSize(2)/height;
    end
    
    % crop
    imgCropped = trainImageData.img( cropMin(2):cropMax(2), cropMin(1):cropMax(1) );
    
    % scale image
    img = imresize(imgCropped, scaleF);
    
    % get new facial points
    newxy = [x y] - repmat( minxy - margin + offset, size(x, 1), 1 );
    newxy = newxy*scaleF;
    
    if 0
        %%% Display image and feature points.
        figure(1);
        imshow(img);
        hold on;
        plot(newxy(:, 1), newxy(:, 2), 'g.');
        plot(newxy(1, 1), newxy(1, 2), 'ro');
    end

end

