function nmx_build()

    cur_dir = pwd;
    cd(fileparts(mfilename('fullpath')));

    % CPU
    fprintf('----- Compiling CPU NMS -----\n');
    mex -O -outdir functions/nms/bin CXXFLAGS="\$CXXFLAGS -std=c++11" -largeArrayDims functions/nms/nms_mex.cpp -output nms_mex;
  
    % GPU
    fprintf('----- Compiling GPU NMS -----\n');
    addpath(fullfile(pwd, 'functions', 'nms'));
    nvmex('functions/nms/nms_gpu_mex.cu', 'functions/nms/bin');
    delete('nms_gpu_mex.o');
    
    addpath('functions/nms/bin');
    
    fprintf('Done.\n');
    
    cd(cur_dir);
    
end
