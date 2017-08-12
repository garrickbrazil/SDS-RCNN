function [image_roidb, bbox_means, bbox_stds] = proposal_prepare_image_roidb_caltech(conf, imdbs, roidbs)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   
    
    if ~iscell(imdbs)
        imdbs = {imdbs};
        roidbs = {roidbs};
    end

    imdbs = imdbs(:);
    roidbs = roidbs(:);
    
    image_roidb = ...
        cellfun(@(x, y) ... // @(imdbs, roidbs)
            arrayfun(@(z) ... //@([1:length(x.image_ids)])
                struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, 'num_classes', x.num_classes, ...
                'boxes', y.rois(z).boxes(y.rois(z).gt, :), 'gt_ignores', y.rois(z).ignores,'class', y.rois(z).class(y.rois(z).gt, :), 'image', [], 'bbox_targets', []), ...
            [1:length(x.image_ids)]', 'UniformOutput', true),...
        imdbs, roidbs, 'UniformOutput', false);
    
    image_roidb = cat(1, image_roidb{:});
    
    % enhance roidb to contain bounding-box regression targets
    [image_roidb, bbox_means, bbox_stds] = append_bbox_regression_targets(conf, image_roidb);
    
end

function [image_roidb, means, stds] = append_bbox_regression_targets(conf, image_roidb)

    num_images = length(image_roidb);
    image_roidb_cell = num2cell(image_roidb, 2);    
    
    % Compute values needed for means and stds
    % var(x) = E(x^2) - E(x)^2
    class_counts = zeros(1, 1) + eps;
    sums = zeros(1, 4);
    squared_sums = zeros(1, 4);
    for i = 1:num_images

        % for fcn, anchors are concated as [channel, height, width], where channel is the fastest dimension.
       [anchors, im_scales] = proposal_locate_anchors(conf, image_roidb_cell{i}.im_size);

       gt_ignores = image_roidb_cell{i}.gt_ignores;

       % add by zhangll, whether the gt_rois empty?
       if isempty(image_roidb_cell{i}.boxes)

           [bbox_targets, ~] = ...
               proposal_compute_targets(conf, image_roidb_cell{i}.boxes, gt_ignores, image_roidb_cell{i}.class,  anchors{1}, image_roidb_cell{i}, im_scales{1});
       else
           [bbox_targets, ~] = ...
               proposal_compute_targets(conf, scale_rois(image_roidb_cell{i}.boxes, image_roidb_cell{i}.im_size, im_scales{1}), gt_ignores, image_roidb_cell{i}.class,  anchors{1}, image_roidb_cell{i}, im_scales{1});
       end
        
        targets = bbox_targets;
        gt_inds = find(targets(:, 1) > 0);
        
        image_roidb(i).has_bbox_target = ~isempty(gt_inds);
        
        if image_roidb(i).has_bbox_target
            class_counts = class_counts + length(gt_inds); 
            sums = sums + sum(targets(gt_inds, 2:end), 1);
            squared_sums = squared_sums + sum(targets(gt_inds, 2:end).^2, 1);
        end
    end

    means = bsxfun(@rdivide, sums, class_counts);
    stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), means.^2)).^0.5;
    
    
end

function scaled_rois = scale_rois(rois, im_size, im_scale)
    im_size_scaled = round(im_size * im_scale);
    scale = (im_size_scaled - 1) ./ (im_size - 1);
    scaled_rois = bsxfun(@times, rois-1, [scale(2), scale(1), scale(2), scale(1)]) + 1;
end
