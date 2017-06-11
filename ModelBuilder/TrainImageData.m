classdef TrainImageData
    
    properties
        filePrefix = '';
        fileImageName = '';
        filePointName = '';
        
        %Image
        img
        
        % original points
        orig_x
        orig_y
        
        % aligned points (after cropping and aligning )
        aligned_x
        aligned_y
        
        % transformation matrix
        align_Tr
    end
    
    methods
    end
    
end

