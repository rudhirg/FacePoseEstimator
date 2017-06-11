function nx = Normalize2D(x)
    maxx = max(max(x));
    minx = min(min(x));

    nx = (x-minx)/(maxx - minx);

