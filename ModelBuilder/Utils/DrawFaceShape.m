function DrawFaceShape(x, y, color)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw face given x, y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

orFace = [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 24 23 22 21 0] + 1;
orEbrowL = [24 23 22 21 26 25 24] + 1;
orEbrowR = [18 19 20 15 16 17 18] + 1;
orEyeL = [27 30 29 31 27 31 29 28 27] + 1;
orEyeR = [34 35 32 36 34 36 32 33 34] + 1;
orNose = [37 38 39 40 46 41 47 42 43 44 45 37] + 1;
orMouth = [48 59 58 57 56 55 54 53 52 50 49 48 60 61 62 63 64 65 48] + 1;

% inverse y for better visual

 
% do visual
if 1
    hold on;
    plot(x(orFace), y(orFace), color);plot(x(orFace), y(orFace), [color '.']);
    %plot(x(orEbrowL), y(orEbrowL), color);
    %plot(x(orEbrowR), y(orEbrowR), color);
    plot(x(orEyeL), y(orEyeL), color);plot(x(orEyeL), y(orEyeL), [color '.']);
    plot(x(orEyeR), y(orEyeR), color);plot(x(orEyeR), y(orEyeR), [color '.']);
    plot(x(orNose), y(orNose), color);plot(x(orNose), y(orNose), [color '.']);
    plot(x(orMouth), y(orMouth), color);plot(x(orMouth), y(orMouth), [color '.']);
    hold off;
end

