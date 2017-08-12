function [input_blobs, random_scale_inds, im_rgb] = proposal_generate_minibatch(conf, image_roidb)
% [input_blobs, random_scale_inds, im_rgb] = proposal_generate_minibatch(conf, image_roidb)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    num_images = length(image_roidb);
    assert(num_images == 1, 'only support num_images == 1');

    % Sample random scales to use for each image in this batch
    random_scale_inds = randi(length(conf.scales), num_images, 1);
    
    rois_per_image = conf.batch_size / num_images;
    fg_rois_per_image = round(conf.batch_size * conf.fg_fraction);
    
    % Get the input image blob
    [im_blob, im_scales, im_rgb] = get_image_blob(conf, image_roidb, random_scale_inds);
    
    rois = image_roidb(1);
    
    % weak segmentation
    if conf.has_weak
        
        ped_mask_weights = single(ones(size(im_blob,1), size(im_blob,2)));
        ped_mask = uint8(zeros(size(im_blob,1), size(im_blob,2)));
        
        for gtind=1:size(rois.boxes,1)
            
            ignore = rois.gt_ignores(gtind);
            gt = rois.boxes(gtind,:);
            
            x1 = min(max(round(gt(1)*im_scales(1)),1),size(ped_mask,2));
            y1 = min(max(round(gt(2)*im_scales(1)),1),size(ped_mask,1));
            x2 = min(max(round(gt(3)*im_scales(1)),1),size(ped_mask,2));
            y2 = min(max(round(gt(4)*im_scales(1)),1),size(ped_mask,1));
            
            w = x2 - x1;
            h = y2 - y1;

            % assign fg label
            ped_mask(y1:y2,x1:x2) = 1;
            
            % cost sensitive
            if conf.cost_sensitive, ped_mask_weights(y1:y2,x1:x2) = single(1 + h/(conf.cost_mean_height*im_scales(1))); end
            
        end
        
        ped_mask = imresize(single(ped_mask), 1/conf.feat_stride, 'nearest');
        ped_mask_weights = imresize(single(ped_mask_weights), 1/conf.feat_stride, 'nearest');
        ped_mask = permute(ped_mask, [2, 1, 3, 4]);
        ped_mask_weights = permute(ped_mask_weights, [2, 1, 3, 4]);
        
    end
    
    % get fcn output size
    img_size = round(image_roidb(1).im_size * im_scales(1));
    output_size = [calc_output_size(img_size(1), conf), calc_output_size(img_size(2), conf)];
    
    % init blobs
    labels_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1), length(image_roidb));
    label_weights_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1), length(image_roidb));
    bbox_targets_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1)*4, length(image_roidb));
    bbox_loss_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1)*4, length(image_roidb));
    
    [labels, label_weights, bbox_targets, bbox_loss] = ...
        sample_rois(conf, image_roidb(1), fg_rois_per_image, rois_per_image, im_scales(1));

    assert(img_size(1) == size(im_blob, 1) && img_size(2) == size(im_blob, 2));

    cur_labels_blob = reshape(labels, size(conf.anchors, 1), output_size(1), output_size(2));
    cur_label_weights_blob = reshape(label_weights, size(conf.anchors, 1), output_size(1), output_size(2));
    cur_bbox_targets_blob = reshape(bbox_targets', size(conf.anchors, 1)*4, output_size(1), output_size(2));
    cur_bbox_loss_blob = reshape(bbox_loss', size(conf.anchors, 1)*4, output_size(1), output_size(2));

    % permute from [channel, height, width], where channel is the
    % fastest dimension to [width, height, channel]
    cur_labels_blob = permute(cur_labels_blob, [3, 2, 1]);

    cur_label_weights_blob = permute(cur_label_weights_blob, [3, 2, 1]);
    cur_bbox_targets_blob = permute(cur_bbox_targets_blob, [3, 2, 1]);
    cur_bbox_loss_blob = permute(cur_bbox_loss_blob, [3, 2, 1]);

    labels_blob(:, :, :, 1) = cur_labels_blob;

    label_weights_blob(:, :, :, 1) = cur_label_weights_blob;
    bbox_targets_blob(:, :, :, 1) = cur_bbox_targets_blob;
    bbox_loss_blob(:, :, :, 1) = cur_bbox_loss_blob;
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = single(permute(im_blob, [2, 1, 3, 4]));
    labels_blob = single(labels_blob);
    labels_blob(labels_blob > 0) = 1;
    label_weights_blob = single(label_weights_blob);
    bbox_targets_blob = single(bbox_targets_blob); 
    bbox_loss_blob = single(bbox_loss_blob);
    
    assert(~isempty(im_blob));
    assert(~isempty(labels_blob));
    assert(~isempty(label_weights_blob));
    assert(~isempty(bbox_targets_blob));
    assert(~isempty(bbox_loss_blob));
    
    input_blobs = {im_blob, labels_blob, label_weights_blob, bbox_targets_blob, bbox_loss_blob};
    
    if conf.has_weak
        input_blobs{length(input_blobs) + 1} = ped_mask;
        input_blobs{length(input_blobs) + 1} = ped_mask_weights;
    end
    
end

%% Build an input blob from the images in the roidb at the specified scales.
function [im_blob, im_scales, im_] = get_image_blob(conf, images, random_scale_inds)
    
    num_images = length(images);
    processed_ims = cell(num_images, 1);
    im_scales = nan(num_images, 1);
    for i = 1:num_images
        im = imread(images(i).image_path);
        im_ = im;
        target_size = conf.scales(random_scale_inds(i));
        
        [im, im_scale] = prep_im_for_blob(im, conf.image_means, target_size, conf.max_size);
        
        im_scales(i) = im_scale;
        processed_ims{i} = im; 
    end
    
    im_blob = im_list_to_blob(processed_ims);
end

%% Generate a random sample of ROIs comprising foreground and background examples.
function [labels, label_weights, bbox_targets, bbox_loss_weights] = sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image, im_scale)

    [anchors, ~] = proposal_locate_anchors(conf, image_roidb.im_size);

    gt_ignores = image_roidb.gt_ignores;

    % add by zhangll, whether the gt_rois empty?
    if isempty(image_roidb.boxes)

       [bbox_targets, ~] = ...
           proposal_compute_targets(conf, image_roidb.boxes, gt_ignores, image_roidb.class,  anchors{1}, image_roidb, im_scale);
    else
       [bbox_targets, ~] = ...
           proposal_compute_targets(conf, scale_rois(image_roidb.boxes, image_roidb.im_size, im_scale), gt_ignores, image_roidb.class,  anchors{1}, image_roidb, im_scale);
    end

    gt_inds = find(bbox_targets(:, 1) > 0);
    if ~isempty(gt_inds)
        bbox_targets(gt_inds, 2:end) = ...
            bsxfun(@minus, bbox_targets(gt_inds, 2:end), conf.bbox_means);
        bbox_targets(gt_inds, 2:end) = ...
            bsxfun(@rdivide, bbox_targets(gt_inds, 2:end), conf.bbox_stds);
    end
    
    ex_asign_labels = bbox_targets(:, 1);

    % Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = find(bbox_targets(:, 1) > 0);
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = find(bbox_targets(:, 1) < 0);
    
    % select foreground
    fg_num = min(fg_rois_per_image, length(fg_inds));
    fg_inds = fg_inds(randperm(length(fg_inds), fg_num));
    
    bg_num = min(rois_per_image - fg_rois_per_image, length(bg_inds));
    bg_inds = bg_inds(randperm(length(bg_inds), bg_num));
    labels = zeros(size(bbox_targets, 1), 1);
    
    % set foreground labels
    labels(fg_inds) = ex_asign_labels(fg_inds);
    assert(all(ex_asign_labels(fg_inds) > 0));
    
    bg_weight = 1;
    label_weights = zeros(size(bbox_targets, 1), 1);
    label_weights(fg_inds) = fg_rois_per_image/fg_num;
    label_weights(bg_inds) = bg_weight;
    
    bbox_targets = single(full(bbox_targets(:, 2:end)));
    
    bbox_loss_weights = bbox_targets * 0;
    bbox_loss_weights(fg_inds, :) = fg_rois_per_image / fg_num;

end

function scaled_rois = scale_rois(rois, im_size, im_scale)
    im_size_scaled = round(im_size * im_scale);
    scale = (im_size_scaled - 1) ./ (im_size - 1);
    scaled_rois = bsxfun(@times, rois-1, [scale(2), scale(1), scale(2), scale(1)]) + 1;
end

