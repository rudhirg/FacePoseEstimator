function [ vect ] = GetXYPointsVector( x, y )
%GETXYPOINTSVECTOR Gets a vector of all face points x,y positions (x1 y1 x2
%y2.... xn yn)
    vect = zeros(1, 2*length(x));
    for i = 1:length(x)
        vect(1, 2*i-1) = x(i);
        vect(1, 2*i) = y(i);
    end

end

