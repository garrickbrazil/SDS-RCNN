function test_rpn(test_prototxt, weights, rpn_conf, anchors, bbox_means, bbox_stds, gpu_id)

    if ~exist('gpu_id',      'var'),  gpu_id       =  1;               end
    
    rpn_conf.gpu_id     = gpu_id;
    rpn_conf.anchors    = anchors;
    rpn_conf.bbox_means = bbox_means;
    rpn_conf.bbox_stds  = bbox_stds;
    
    warning('off', 'MATLAB:class:DestructorError');
    
    fprintf('Processing test.. ');

    reset_caffe(rpn_conf);
    
    rpn_conf.test_dir     = [pwd '/datasets/' rpn_conf.dataset_test  '/test'];
    
    % net
    results_dir = [pwd '/.tmpresults'];

    % test net
    net = caffe.Net(test_prototxt, 'test');
    net.copy_from([weights]);

    % evaluate
    [mr, recall] = evaluate_results_rpn(rpn_conf, net, results_dir, rpn_conf.test_dir, rpn_conf.test_db);
    fprintf('mr %.4f, recall %.4f\n', mr, recall);

    reset_caffe(rpn_conf);
    
    if (exist(results_dir, 'dir')), rmdir(results_dir, 's'); end

end