function startup()
% startup()
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    curdir = fileparts(mfilename('fullpath'));
    addpath(fullfile(curdir, 'functions', 'nms'));
    addpath(fullfile(curdir, 'functions', 'rcnn'));
    addpath(fullfile(curdir, 'functions', 'rpn'));
    addpath(fullfile(curdir, 'functions', 'utils'));
    addpath(fullfile(curdir, 'functions', 'nms', 'bin'));
    addpath(fullfile(curdir, 'experiments'));
    
    addpath(fullfile(curdir, 'external/caffe/matlab/'));
    addpath(fullfile(curdir, 'external', 'caltech_toolbox'));
    addpath(genpath(fullfile(curdir, 'external', 'pdollar_toolbox')));
    
end
