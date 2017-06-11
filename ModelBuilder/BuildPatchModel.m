function patchModel = BuildPatchModel( trainImageDataList, params )
    numTrainData = length(trainImageDataList);
    numPatches = size(trainImageDataList(1).aligned_x, 2);
    
    featLen = params.PatchSize(1)*params.PatchSize(2);
    
    genSVMData = true;
    
    % see if svm data already exists
    svmDataDir = '../data/svm-data';
    Exist = dir(svmDataDir);
    if ~isempty(Exist)
        genSVMData = false;
    end
    
    if genSVMData == true
        if params.debugPatches == 1
            ParentDir = '../templates/';

            % create patch dirs.
            mkdir(ParentDir);
            for i = 1:numPatches
                dirName = sprintf([ParentDir '/%d'], i);
                Exist = dir(dirName);
                if(isempty(Exist))
                    mkdir(dirName);
                    mkdir([dirName, '/pos']);
                    mkdir([dirName, '/neg']);
                end
            end  
        end


        % for each patch
        for p = 1:numPatches
           posPatches = zeros( numTrainData*20, params.PatchSize(1)*params.PatchSize(2));
           negPatches = zeros( numTrainData*120, params.PatchSize(1)*params.PatchSize(2));

           posCnt = 0;
           negCnt = 0;

           % for each training image
           for i = 1:numTrainData
              % get positive patches
              patchXY = [trainImageDataList(i).aligned_x(p) trainImageDataList(i).aligned_y(p)];
              patches = GetPosPatch( trainImageDataList(i).img,  patchXY(1), patchXY(2), params );

              pLen = size(patches, 1);
              posPatches(posCnt+1: posCnt+pLen, :) = patches;
              posCnt = posCnt + pLen;

              % for debug purpose
              if params.debugPatches
                szx = size(patches, 1);
                for kk = 1:szx
                    % write file
                    fname = sprintf([ParentDir '/%d/pos/%d_%d.bmp'], p, i, kk);
                    twr = patches(kk, :);
                    twr = reshape(twr, params.PatchSize);

                    imwrite(twr, fname, 'bmp');
                end
              end

              patches = GetNegPatch( trainImageDataList(i).img,  patchXY(1), patchXY(2), params );

              pLen = size(patches, 1);
              negPatches(negCnt+1: negCnt+pLen, :) = patches;
              negCnt = negCnt + pLen;

              % for debug purpose
              if params.debugPatches
                szx = size(patches, 1);
                for kk = 1:szx
                    % write file
                    fname = sprintf([ParentDir '/%d/neg/%d_%d.bmp'], p, i, kk);
                    twr = patches(kk, :);
                    twr = reshape(twr, params.PatchSize);

                    imwrite(twr, fname, 'bmp');
                end
              end

           end

            % c. save pos and neg data
            mkdir( '../data/svm-data' );
            fname = sprintf('../data/svm-data/pos%d.mat', p);
            pat = posPatches(1:posCnt, :);
            save(fname, 'pat');

            fname = sprintf('../data/svm-data/neg%d.mat', p);
            pat = negPatches(1:negCnt, :);
            save(fname, 'pat');

            posPatches = 0;
            negPatches = 0;

            disp(p);

        end
    end
    
    %% Train SVM
    for i=1:numPatches
        fname = sprintf('../data/svm-data/pos%d.mat', i);
        posPatches = load(fname);

        fname = sprintf('../data/svm-data/neg%d.mat', i);
        negPatches = load(fname);

        % Assemble data
        numpos = size(posPatches.pat, 1);
        numneg = size(negPatches.pat, 1);
        
        if numpos == 0 || numneg == 0
            Weights(i, :) = zeros(1, featLen);
            continue;
        end

        % get the best c
        best_c = 1e-5;
%         best_acc = -1.0;
%         for t=0:4
%             c = 0.001*10^t;
%             options = sprintf('-t 0 -c %f -v 5', c);
%             model = svmtrain([ones(numpos, 1); -ones(numneg, 1)], [posPatches.pat; negPatches.pat],options);
%             disp( ['c:', num2str(c), ' acc:', num2str(model)] );
%             if model > best_acc
%                 best_c = c;
%                 best_acc = model;
%             end
%         end
        
        disp( ['best_c: ', num2str(best_c)] );
        options = sprintf('-t 0 -c %f', best_c);
        model = svmtrain([ones(numpos, 1); -ones(numneg, 1)], [posPatches.pat; negPatches.pat],options);
        
        sss = sprintf('done %d', i);
        disp(sss);

        Weights(i, :) = (model.sv_coef' * full(model.SVs));
        Rhos(i) = model.rho;

        ShowSVMWeights = true;
        if(ShowSVMWeights)
            ww = reshape(Weights(i, :), params.PatchSize);
            kw = Normalize2D(ww);
            figure(96);imshow(kw, []);
            title(sprintf('point %d', i-1));
            wname = sprintf('../data/svm-data/weight_%d.bmp', i-1);
            imwrite(kw, wname);
        end

    end

    % Store the patch
    patchModel.NumTrainingSamples = numTrainData;
    patchModel.NumPatches = numPatches;
    patchModel.PatchSize = params.PatchSize;
    patchModel.CanvasSize = params.CanvasSize;
    patchModel.Weights = Weights;
    patchModel.rho = Rhos;
    
end

