function conf = config_func()
    
    conf.model                  =  'VGG16_weak_seg';
    conf.dataset_train          =  'caltechx10';
    conf.dataset_test           =  'caltechx1';
    conf.dataset_val            =  'caltechval';
    conf.ext                    =  '.jpg';
    
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
    conf.fg_thresh              =  0.7;
    conf.image_means            = [123.6800, 116.7790, 103.9390];

    % network settings
    conf.train_batch_size       =  20;        % number of proposals train
    conf.test_batch_size        =  15;        % number of proposals test
    conf.crop_size              =  [112 112]; % size of images
    conf.has_weak               =  true;      % has weak segmentation?
    conf.weak_seg_crop          =  [7 7];     % weak segmentation size
    conf.feat_stride            =  16;        % network stride
    conf.cost_sensitive         =  true;      % use cost sensitive
    conf.cost_mean_height       =  50;        % cost sensitive mean
    conf.fg_image_ratio         =  0.5;       % percent fg images
    conf.batch_size             =  120;       % number fg boxes
    conf.natural_fg_weight      =  true;      % ignore fg_fraction!
    conf.fg_fraction            =  1/5;       % percent fg boxes
    conf.feat_scores            =  true;      % fuse feature scores of rpn
    conf.padfactor              =  0.2;       % percent padding
    
    %% testing
    conf.test_db                = 'UsaTest';     % dataset to test with
    conf.val_db                 = 'UsaTrainVal'; % dataset to test with
    conf.min_gt_height          =  30;           % smallest gt to train on
    conf.test_min_h             =  50;           % database setting for min gt
    
    conf.image_means = reshape(conf.image_means, [1 1 3]);

end
