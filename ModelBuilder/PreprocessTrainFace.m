function [ newxy, newImage ] = PreprocessTrainFace( firstFace, trainImageData, Params )

    cvw = Params.CanvasSize(1);
    cvh = Params.CanvasSize(2);

    basexy = [firstFace.aligned_x' firstFace.aligned_y'];

    xy = [trainImageData.orig_x' trainImageData.orig_y'];

    % Use procrustes analysis
    % to align shape.
    % gets the transformation from xy to basexy
    [d z tform] = procrustes(basexy, xy, 'Reflection',false);

    % Crop image
    % [NOTE]: we are trying to find the inverse transformation of tform
    % in affine transformation inv(Rot) = R', inv(Scale) = 1/scale
    % and inv(Translation) can be found out by:
    % x_new = sRx + T
    % x = (x_new - T)/sR
    % x = (x_new/s)*R' - (1/s)*T*R' 
    % thus T_new = (1/s)*T*R' (thats what the below eq is to find
    % translation
    Translatexy = -1/tform.b*tform.c*tform.T';
    Translatexy = Translatexy(1, :);


    transM = [1/tform.b*tform.T Translatexy'];
    cvXY = [1 cvw 1 cvw;
        1 1 cvh cvh];

    % Quad2Box gets the new Image (NI) such that: transM( NI ) = I
    newImage = Quad2Box(trainImageData.img, cvXY, transM);
    newxy = z;

    if 0
        figure(129);
        imshow(newImage);
        hold on;
        plot(z(:, 1), z(:, 2), 'r.');
    end
end

