function neg = GetNegPatch( img, xc, yc, params )
    %makes positive patch sample
    w = params.PatchSize(1);
    h = params.PatchSize(2);
    
    % stay away from central point by 1/2th of width
    offset = w; 
    
    xb = xc - w/2;
    yb = yc - h/2;
    
    neg = zeros(30, w*h);
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
            
            % skip center point, within some error tolerance.
            % center points belongs to positive examples.
            if(abs(x - xc + w/2)<4 && abs(y - yc + w/2)<4) 
                continue;
            end

            subImg = img(y:y+h-1, x:x+w-1);
            subImg = Normalize2D( subImg );
            vec = reshape(subImg, numel(subImg), 1);
            
            cnt = cnt + 1;
            neg(cnt, :) = vec';
            
        end
    end

    neg = neg(1:cnt, :);
    
    % Add some uniform patch as
    % negative training image.
    neg(cnt+1, :) = zeros(1, w*h);
    neg(cnt+2, :) = ones(1, w*h);
    neg(cnt+3, :) = 0.5*ones(1, w*h);
    neg(cnt+4, :) = 0.25*ones(1, w*h);
    neg = neg(1:cnt+4, :);
end

