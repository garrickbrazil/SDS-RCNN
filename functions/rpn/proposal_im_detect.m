function [pred_boxes, scores, feat_scores_bg, feat_scores_fg] = proposal_im_detect(conf, caffe_net, im)
% [pred_boxes, scores, feat_scores_bg, feat_scores_fg] = proposal_im_detect(conf, caffe_net, im)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------    

    im = single(im);
    [im_blob, im_scales] = prep_im_for_blob(im, conf.image_means, conf.scales, conf.max_size);
    im_size = size(im);
    scaled_im_size = round(im_size * im_scales);
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg    
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);
    
    net_inputs = {im_blob};

    % Reshape net's input blobs
    caffe_net = reshape_input_data(caffe_net, net_inputs);
    caffe_net.forward(net_inputs);
        
    % Apply bounding-box regression deltas
    box_deltas = caffe_net.blobs('proposal_bbox_pred').get_data();
    featuremap_size = [size(box_deltas, 2), size(box_deltas, 1)];
    
    % permute from [width, height, channel] to [channel, height, width], where channel is the fastest dimension
    box_deltas = permute(box_deltas, [3, 2, 1]);
    box_deltas = reshape(box_deltas, 4, [])';
    
    anchors = proposal_locate_anchors(conf, size(im), conf.scales, featuremap_size);
    pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_deltas);
    
    % scale back
    pred_boxes = bsxfun(@times, pred_boxes - 1, ...
        ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
    pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));
    
    % use softmax estimated probabilities
    scores = caffe_net.blobs('proposal_cls_prob').get_data();
    scores = scores(:, :, end);
    scores = reshape(scores, size(caffe_net.blobs('proposal_bbox_pred').get_data(), 1), size(caffe_net.blobs('proposal_bbox_pred').get_data(), 2), []);
    
    % store features
    feat_scores = caffe_net.blobs('proposal_cls_score_reshape').get_data();
    feat_scores_bg = feat_scores(:, :, 1);
    feat_scores_fg = feat_scores(:, :, 2);
    
    feat_scores_fg = reshape(feat_scores_fg, size(caffe_net.blobs('proposal_bbox_pred').get_data(), 1), size(caffe_net.blobs('proposal_bbox_pred').get_data(), 2), []);
    feat_scores_fg = permute(feat_scores_fg, [3, 2, 1]);
    feat_scores_fg = feat_scores_fg(:);
    feat_scores_bg = reshape(feat_scores_bg, size(caffe_net.blobs('proposal_bbox_pred').get_data(), 1), size(caffe_net.blobs('proposal_bbox_pred').get_data(), 2), []);
    feat_scores_bg = permute(feat_scores_bg, [3, 2, 1]);
    feat_scores_bg = feat_scores_bg(:);
    
    % permute from [width, height, channel] to [channel, height, width], where channel is the
        % fastest dimension
    scores = permute(scores, [3, 2, 1]);
    scores = scores(:);
    
    % drop too small boxes
    [pred_boxes, scores, valid_ind] = filter_boxes(conf.test_min_box_size, conf.test_min_box_height, pred_boxes, scores);
    
    % sort
    [scores, scores_ind] = sort(scores, 'descend');
    pred_boxes = pred_boxes(scores_ind, :);
    
    feat_scores_fg = feat_scores_fg(valid_ind, :);
    feat_scores_fg = feat_scores_fg(scores_ind, :);
    feat_scores_bg = feat_scores_bg(valid_ind, :);
    feat_scores_bg = feat_scores_bg(scores_ind, :);
    
end

function [boxes, scores, valid_ind] = filter_boxes(min_box_size, min_box_height, boxes, scores)
    widths = boxes(:, 3) - boxes(:, 1) + 1;
    heights = boxes(:, 4) - boxes(:, 2) + 1;
    
    valid_ind = widths >= min_box_size & heights >= min_box_size & heights >= min_box_height;
    boxes = boxes(valid_ind, :);
    scores = scores(valid_ind, :);
end
    
function boxes = clip_boxes(boxes, im_width, im_height)
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end
    
