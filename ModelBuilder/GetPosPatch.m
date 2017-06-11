function pos = GetPosPatch( img, xc, yc, params )
    %makes positive patch sample
    w = params.PatchSize(1);
    h = params.PatchSize(2);
    
    % allow 1/8th of width tolerance
    offset = round(w/8); 
    
    xb = xc - w/2;
    yb = yc - h/2;
    
    pos = zeros(30, w*h);
    cnt = 0;
    
    for x = xb - offset: offset : xb + offset + 2
        % out of boundary
        if x < 1 || x+w > params.CanvasSize(1)
            continue;
        end
        
        for y = yb - offset: offset : yb + offset + 2
           % out of boundary
            if y < 1 || y+h > params.CanvasSize(2)
                continue;
            end 
            
            subImg = img(y:y+h-1, x:x+w-1);
            subImg = Normalize2D( subImg );
            vec = reshape(subImg, numel(subImg), 1);
            
            cnt = cnt + 1;
            pos(cnt, :) = vec';
            
        end
    end

    pos = pos(1:cnt, :);
end

