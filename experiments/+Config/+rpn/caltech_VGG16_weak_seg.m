function conf = config_func()
    
    conf.model                  =  'VGG16_weak_seg';
    conf.dataset_train          =  'caltechx10';
    conf.dataset_test           =  'caltechx1';
    conf.dataset_val            =  'caltechval';
    
    % solver
    conf.solver_type            =  'SGD';
    conf.lr                     =  0.001;
    conf.step_size              =  60000;
    conf.max_iter               =  120000;
    conf.snapshot_iter          =  10000;

    % general    
    conf.display_iter           =  1000;
    conf.rng_seed               =  3;
    conf.mat_rng_seed           =  3;
    conf.scales                 =  720;
    conf.max_size               =  1800;
    conf.bg_thresh_hi           =  0.5;
    conf.bg_thresh_lo           =  0;
    conf.fg_thresh              =  0.5;    
    conf.pretrained             =  'vgg16.caffemodel';
    conf.image_means            = [123.6800, 116.7790, 103.9390];

    % network settings    
    conf.has_weak               =  true; % has weak segmentation?
    conf.feat_stride            =  16;   % network stride
    conf.cost_sensitive         =  true;      % use cost sensitive
    conf.cost_mean_height       =  50;   % cost sensitive mean
    conf.fg_image_ratio         =  0.5;  % percent fg images
    conf.batch_size             =  120;  % number fg boxes
    conf.fg_fraction            =  1/5;  % percent fg boxes

    % anchors
    conf.anchor_scales          =  1.6*(1.385.^(0:8));
    conf.anchor_ratios          =  0.41;
    conf.base_anchor_size       =  16;
    
    %% testing
    conf.test_min_box_height    =  50;           % min box height to keep
    conf.test_min_box_size      =  16;           % min box size to keep (w || h)
    conf.nms_per_nms_topN       =  10000;        % boxes before nms
    conf.nms_overlap_thres      =  0.5;          % nms threshold IoU
    conf.nms_after_nms_topN     =  40;           % boxes after nms
    conf.test_db                = 'UsaTest';     % dataset to test with
    conf.val_db                 = 'UsaTrainVal'; % dataset to test with
    conf.min_gt_height          =  30;           % smallest gt to train on
    conf.test_min_h             =  50;           % database setting for min gt
    
    conf.image_means = reshape(conf.image_means, [1 1 3]);

end
