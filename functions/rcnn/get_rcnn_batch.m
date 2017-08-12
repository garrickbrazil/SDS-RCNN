function net_inputs = get_rcnn_batch(conf, rois, phase)
    
    training = strcmp(phase, 'train');
    
    if training
        set = conf.dataset_train;
        batch_size = conf.train_batch_size;
        im_dir = conf.train_dir;
    elseif strcmp(phase, 'val');
        set = conf.dataset_val; 
        batch_size = conf.test_batch_size;
        im_dir = conf.val_dir;
    else
        set = conf.dataset_test; 
        batch_size = conf.test_batch_size;
        im_dir = conf.test_dir;
    end

    im = (imread([im_dir '/images/' rois.image_id conf.ext]));
    impadH = round(size(im,1)*conf.padfactor);
    impadW = round(size(im,2)*conf.padfactor);
    im = padarray(im, [impadH impadW]);
    
    rois_boxes = rois.boxes;
    rois_batch = single(zeros([conf.crop_size(2) conf.crop_size(1) 3 batch_size]));
    feat_scores_fg = rois.feat_scores_fg;
    feat_scores_bg = rois.feat_scores_bg;
    
    if training
        
        % paint weak seg gt
        if conf.has_weak
            
            ped_mask = uint8(zeros(size(im,1), size(im,2))); 
            
            for gtind=1:size(rois.gts,1)

                gt = rois.gts(gtind,:);
                
                x1 = min(max(round(gt(1)),1),size(ped_mask,2));
                y1 = min(max(round(gt(2)),1),size(ped_mask,1));
                x2 = min(max(round(gt(3)),1),size(ped_mask,2));
                y2 = min(max(round(gt(4)),1),size(ped_mask,1));

                ped_mask(y1:y2,x1:x2) = 1;

            end

            ped_mask = padarray(ped_mask, [impadH impadW]);
            
            weak_seg_batch = single(zeros([conf.weak_seg_crop(2) conf.weak_seg_crop(1) 1 batch_size])); 
            weak_seg_weights_batch = single(zeros([conf.weak_seg_crop(2) conf.weak_seg_crop(1) 1 batch_size]));
            
        end
    
        rois_labels = rois.ols >= conf.fg_thresh;
        cost_weights = zeros([batch_size,1]); 
    end
    
    for j=1:batch_size

        % get box info
        x1 = rois_boxes(j, 1);
        y1 = rois_boxes(j, 2);
        x2 = rois_boxes(j, 3);
        y2 = rois_boxes(j, 4);
        w = x2-x1;
        h = y2-y1;
        
        x1 = x1 - w*conf.padfactor + impadW;
        y1 = y1 - h*conf.padfactor + impadH;
        w = w + w*conf.padfactor;
        h = h + h*conf.padfactor;
                
        % crop and resize proposal
        propim = imcrop(im, [x1 y1 w h]);
        propim = imresize(single(propim), [conf.crop_size(1) conf.crop_size(2)]);
        propim = bsxfun(@minus, single(propim), conf.image_means);
        
        % permute data into caffe c++ memory, thus [num, channels, height, width]
        propim = propim(:, :, [3, 2, 1], :);
        propim = permute(propim, [2, 1, 3, 4]);
        rois_batch(:,:,:,j) = single(propim);
        
        if training 
        
            cost_weights(j) = h/conf.cost_mean_height; 
            
            if conf.has_weak
                prop_mask = imcrop(ped_mask, [x1 y1 w h]);
                prop_mask = imresize(single(prop_mask), [conf.weak_seg_crop(1) conf.weak_seg_crop(2)], 'nearest');
                prop_mask = permute(prop_mask, [2, 1, 3, 4]);
                weak_seg_batch(:,:,:,j) = single(prop_mask);
            end
        end
        
    end
    
    if training
        
        rois_labels = single(rois_labels(1:batch_size));
        rois_label_weights = ones([size(rois_labels),1]);

        rois_fg = rois_labels == 1;
        
        fg_rois_per_image = round(batch_size * conf.fg_fraction);
        
        if ~conf.natural_fg_weight && sum(rois_fg(:)) > 0, 
            rois_label_weights(rois_fg) = fg_rois_per_image/sum(rois_fg(:)); 
        end
        
        if conf.cost_sensitive
            rois_label_weights = rois_label_weights + cost_weights;
        end
        
        rois_labels = single(permute(rois_labels, [3, 4, 2, 1]));
        rois_label_weights = single(permute(rois_label_weights, [3, 4, 2, 1]));
        net_inputs = {rois_batch, rois_labels, rois_label_weights};
       
        if conf.has_weak
            
            for weakind=1:batch_size
                weak_seg_weights_batch(:,:,:,weakind) = rois_label_weights(weakind);
            end
            
            net_inputs{length(net_inputs) + 1} = weak_seg_batch;
            net_inputs{length(net_inputs) + 1} = weak_seg_weights_batch;
        end
        
    % testing
    else
        net_inputs = {rois_batch};
    end
    
    
    if conf.feat_scores
        rois_feat_scores = [feat_scores_bg feat_scores_fg];
        rois_feat_scores = single(rois_feat_scores(1:batch_size, :));
        rois_feat_scores = single(permute(rois_feat_scores, [3, 4, 2, 1]));
        net_inputs{length(net_inputs) + 1} = rois_feat_scores;
    end

end