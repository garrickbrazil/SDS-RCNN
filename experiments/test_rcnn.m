function test_rcnn(rpn_prototxt, rpn_weights, rpn_conf, anchors, bbox_means, bbox_stds, rcnn_prototxt, rcnn_weights, rcnn_conf, gpu_id)

    if ~exist('gpu_id',      'var'),  gpu_id       =  1;               end
    
    rpn_conf.gpu_id     = gpu_id;
    rpn_conf.anchors    = anchors;
    rpn_conf.bbox_means = bbox_means;
    rpn_conf.bbox_stds  = bbox_stds;
    
    warning('off', 'MATLAB:class:DestructorError');
    
    fprintf('Processing test.. ');

    reset_caffe(rpn_conf);
    
    test_dir     = [pwd '/datasets/' rpn_conf.dataset_test  '/test'];
    results_dir = [pwd '/.tmpresults'];
    
    if (exist(results_dir, 'dir')), rmdir(results_dir, 's'); end

    rpn_net = caffe.Net(rpn_prototxt, 'test');
    rpn_net.copy_from([rpn_weights]);
    
    rcnn_net = caffe.Net(rcnn_prototxt, 'test');
    rcnn_net.copy_from([rcnn_weights]);
    
    imlist = dir([test_dir '/images/*.jpg']);
    
    for imind=1:length(imlist)
        
        %imobj = imlist{imind};
        im = imread([test_dir '/images/' imlist(imind).name]);
        
        [boxes, scores, feat_scores_bg, feat_scores_fg] = proposal_im_detect(rpn_conf, rpn_net, im);
        
        % filter rpn
        proposal_num = rcnn_conf.test_batch_size;
        [aboxes, inds] = nms_filter([boxes, scores], rpn_conf.nms_per_nms_topN, rpn_conf.nms_overlap_thres, proposal_num, 1);
        
        boxes = aboxes(:, 1:4);
        scores = aboxes(:, 5);
        
        feat_scores_bg = feat_scores_bg(inds,:);
        feat_scores_fg = feat_scores_fg(inds,:);
        
        feat_scores_bg = feat_scores_bg(1:min(length(aboxes), proposal_num), :);
        feat_scores_fg = feat_scores_fg(1:min(length(aboxes), proposal_num), :);
        
        impadH = round(size(im,1)*rcnn_conf.padfactor);
        impadW = round(size(im,2)*rcnn_conf.padfactor);
        im = padarray(im, [impadH impadW]);

        rois_batch = single(zeros([rcnn_conf.crop_size(2) rcnn_conf.crop_size(1) 3 proposal_num]));
        
        for j=1:proposal_num

            % get box info
            x1 = boxes(j, 1);
            y1 = boxes(j, 2);
            x2 = boxes(j, 3);
            y2 = boxes(j, 4);
            w = x2-x1;
            h = y2-y1;

            x1 = x1 - w*rcnn_conf.padfactor + impadW;
            y1 = y1 - h*rcnn_conf.padfactor + impadH;
            w = w + w*rcnn_conf.padfactor;
            h = h + h*rcnn_conf.padfactor;

            % crop and resize proposal
            propim = imcrop(im, [x1 y1 w h]);
            propim = imresize(single(propim), [rcnn_conf.crop_size(1) rcnn_conf.crop_size(2)]);
            propim = bsxfun(@minus, single(propim), rcnn_conf.image_means);

            % permute data into caffe c++ memory, thus [num, channels, height, width]
            propim = propim(:, :, [3, 2, 1], :);
            propim = permute(propim, [2, 1, 3, 4]);
            rois_batch(:,:,:,j) = single(propim);

        end
        
        net_inputs = {rois_batch};
        rois_feat_scores = [feat_scores_bg feat_scores_fg];
        rois_feat_scores = single(rois_feat_scores(1:proposal_num, :));
        rois_feat_scores = single(permute(rois_feat_scores, [3, 4, 2, 1]));
        net_inputs{length(net_inputs) + 1} = rois_feat_scores;
        
        rcnn_net = reshape_input_data(rcnn_net, net_inputs);
        rcnn_net.forward(net_inputs);
        
        [~,image_id] = fileparts(imlist(imind).name);
        
        reg = regexp(image_id, '(set\d\d)_(V\d\d\d)_I(\d\d\d\d\d)', 'tokens');
        setname = reg{1}{1};
        vname = reg{1}{2};
        iname = reg{1}{3};
        inum = str2num(iname) + 1;

        mkdir_if_missing([results_dir '/' setname]);

        fid  = fopen([results_dir '/' setname '/' vname '.txt'], 'a');

        cls_scores_fused = rcnn_net.blobs('cls_score_sm').get_data();
        cls_scores_fused = cls_scores_fused(end,:);
        cls_scores_fused = cls_scores_fused(:);

        % score 1 (fused)
        aboxes = [boxes, cls_scores_fused];

        for scoreind=1:size(aboxes,1)

            x1 = aboxes(scoreind, 1);
            y1 = aboxes(scoreind, 2);
            x2 = aboxes(scoreind, 3);
            y2 = aboxes(scoreind, 4);
            score = aboxes(scoreind, 5);

            w = x2 - x1;
            h = y2 - y1;

            fprintf(fid, '%d,%.3f,%.3f,%.3f,%.3f,%.6f\n', [inum x1 y1 w h score]);

        end
        
        fclose(fid);
        
    end
    
    mr = evaluate_result_dir({results_dir}, rcnn_conf.test_db, rcnn_conf.test_min_h);
    
    fprintf('MR=%.4f\n', mr);
    
    reset_caffe(rpn_conf);
    
    if (exist(results_dir, 'dir')), rmdir(results_dir, 's'); end

end