function [ x y ] = ReadImagePoints( fileName, folderPath )
    filePath = [folderPath '/' fileName];
    fd = fopen( filePath );
    
    % version
    fgetl( fd );
    
    % get num points
    line = fgetl( fd );
    
    %bracket
    line = fgetl(fd);

    %pts
    tempxy = [];
    for i=1:68
        [dat, cnt] = fscanf(fd, '%f %f\n', 2);
        tempxy = [tempxy dat];
    end

    x = tempxy(1, :);
    y = tempxy(2, :);

    fclose(fd);

end

