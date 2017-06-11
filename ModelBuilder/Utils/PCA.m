function [ eigenVect, eigenVal, mat_mean ] = PCA( mat )
%PCA 
    sz = size( mat, 2);
    % calculate mean
    mat_mean = sum( mat, 2 )/sz;
    
    % Substract the mean
    mat_normalized = (mat-repmat(mat_mean,1,sz))/ sqrt(sz-1);

    % pca
    [eigenVect, eigenVal] = svds( mat_normalized, sz );
    eigenVal = diag(eigenVal).^2;
    
end

