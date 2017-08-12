function [aboxes, valid] = nms_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    
    if per_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        valid = nms(aboxes, nms_overlap_thres, use_gpu);
        aboxes = aboxes(valid, :);       
    end
    
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
        valid = valid(1:min(length(aboxes), after_nms_topN));
    end
end
