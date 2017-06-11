function [ mat_pts ] = GetFacePointsMatrix( x, y )
%GETFACEPOINTSMATRIX Takes the x axis points, y axis points and returns
%matrix in row form
    mat_pts = zeros(length(x), 2)
    for i = 1: length(x)
       mat_pts(i,1) = x(i);
       mat_pts(i,2) = y(i);
    end
end

