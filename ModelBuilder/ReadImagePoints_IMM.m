function [ x y ] = ReadImagePoints_IMM( fileName, folderPath )
    imgWid = 640;
    imgHt = 480;
    
    filePath = [folderPath '/' fileName];
    fd = fopen( filePath );
    
    % loop to get rid of all comments
    while true
        line = fgetl( fd );
        if isempty( line ) == false && line(1) ~= '#' 
            break;
        end
    end
    
    % num points
    numPoints = str2num( line );
    
    %empty line
    fgetl( fd );

    % loop to get rid of all comments
    while true
        line = fgetl( fd );
        if line ~= '#' 
            break;
        end
    end

    %pts
    tempxy = [];
    for i=1:numPoints
        data = sscanf(line, '%d %d %f %f %d %d %d\n', 7);
        tempxy = [tempxy [data(3)*imgWid; data(4)*imgHt]];
        line = fgetl( fd );
    end

    x = tempxy(1, :);
    y = tempxy(2, :);

    fclose(fd);

end

