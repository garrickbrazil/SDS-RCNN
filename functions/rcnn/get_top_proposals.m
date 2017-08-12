function [rois] = get_top_proposals(conf, rpn_net, roidb, imdb)

    im_dir = imdb.image_dir;
    im_count = length(imdb.image_ids);
    rois = cell([im_count 1]);
    
    tic;

    for i=1:im_count
       
        im_path = [im_dir '/' imdb.image_ids{i} '.jpg'];
        gt = roidb.rois(i);
        
        im = imread(im_path);
        
        [boxes, scores, feat_scores_bg, feat_scores_fg] = proposal_im_detect(conf, rpn_net, im);
        
        % filter rpn
        proposal_num = 120;
        [aboxes, inds] = nms_filter([boxes, scores], conf.nms_per_nms_topN, conf.nms_overlap_thres, proposal_num, 1);
        
        boxes = aboxes(:, 1:4);
        scores = aboxes(:, 5);
        
        feat_scores_bg = feat_scores_bg(inds,:);
        feat_scores_fg = feat_scores_fg(inds,:);
        
        feat_scores_bg = feat_scores_bg(1:min(length(aboxes), proposal_num), :);
        feat_scores_fg = feat_scores_fg(1:min(length(aboxes), proposal_num), :);
        
        if size(gt.boxes,1) > 0
    
            ols = boxoverlap(boxes, gt.boxes);
            [max_ols, targets] = max(ols, [], 2);
            labels = max_ols > 0.5;

        else 
            labels = logical(zeros([size(boxes,1),1]));
            targets = zeros([size(boxes,1),1]);
            max_ols = zeros([size(boxes,1),1]);
        end
        
        rois{i}.image_id = imdb.image_ids{i};
        rois{i}.boxes = single(boxes);
        rois{i}.labels = labels;
        rois{i}.targets = uint8(targets);
        rois{i}.gts = single(gt.boxes);
        rois{i}.ignores = gt.ignores;
        rois{i}.scores = single(scores);
        rois{i}.ols = single(max_ols);
        rois{i}.feat_scores_bg = single(feat_scores_bg);
        rois{i}.feat_scores_fg = single(feat_scores_fg);
        
        if mod(i, 1000) == 0
            dt = toc/(i);
            timeleft = dt*(im_count - i);
            if timeleft > 3600, timeleft = sprintf('%.1fh', timeleft/3600);
            elseif timeleft > 60, timeleft = sprintf('%.1fm', timeleft/60);
            else timeleft = sprintf('%.1fs', timeleft); end
            fprintf('%d/%d, dt=%.4f, eta=%s\n', i, im_count, dt, timeleft);
        end
    end
end

