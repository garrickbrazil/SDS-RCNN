function [shuffled_inds, sub_inds] = proposal_generate_batch(shuffled_inds, image_roidb, ims_per_batch, fg_image_ratio)

    % shuffle training data per batch
    if isempty(shuffled_inds)
        
        if ims_per_batch == 1
            
            empty_image_inds = arrayfun(@(x) x.has_bbox_target, image_roidb, 'UniformOutput', true);
            nonempty_image_inds = ~empty_image_inds;
            empty_image_inds = find(empty_image_inds);
            nonempty_image_inds = find(nonempty_image_inds);
            
            if fg_image_ratio == 1
                shuffled_inds = nonempty_image_inds;
            else
                if length(nonempty_image_inds) > length(empty_image_inds)
                    empty_image_inds = repmat(empty_image_inds, ceil(length(nonempty_image_inds) / length(empty_image_inds)), 1);
                    empty_image_inds = empty_image_inds(1:length(nonempty_image_inds));
                else
                    nonempty_image_inds = repmat(nonempty_image_inds, ceil(length(empty_image_inds) / length(nonempty_image_inds)), 1);
                    nonempty_image_inds = nonempty_image_inds(1:length(empty_image_inds));
                end
                empty_image_inds = empty_image_inds(randperm(length(empty_image_inds), round(length(empty_image_inds) * (1 - fg_image_ratio))));
                nonempty_image_inds = nonempty_image_inds(randperm(length(nonempty_image_inds), round(length(nonempty_image_inds) * fg_image_ratio)));
                
                shuffled_inds = [nonempty_image_inds; empty_image_inds];
            end
            
            shuffled_inds = shuffled_inds(randperm(size(shuffled_inds, 1)));
            shuffled_inds = num2cell(shuffled_inds, 2);
            
        else
            
            % make sure each minibatch, contain half (or half+1) gt-nonempty
            % image, and half gt-empty image
            empty_image_inds = arrayfun(@(x) sum(x.bbox_targets{1}(:, 1)==1) == 0, image_roidb, 'UniformOutput', true);
            nonempty_image_inds = ~empty_image_inds;
            empty_image_inds = find(empty_image_inds);
            nonempty_image_inds = find(nonempty_image_inds);
            
            empty_image_per_batch = floor(ims_per_batch / 2);
            nonempty_image_per_batch = ceil(ims_per_batch / 2);
            
            % random perm
            lim = floor(length(nonempty_image_inds) / nonempty_image_per_batch) * nonempty_image_per_batch;
            nonempty_image_inds = nonempty_image_inds(randperm(length(nonempty_image_inds), lim));
            nonempty_image_inds = reshape(nonempty_image_inds, nonempty_image_per_batch, []);
            if numel(empty_image_inds) >= lim
                empty_image_inds = empty_image_inds(randperm(length(nonempty_image_inds), empty_image_per_batch*lim/nonempty_image_per_batch));
            else
                empty_image_inds = empty_image_inds(mod(randperm(lim, empty_image_per_batch*lim/nonempty_image_per_batch), length(empty_image_inds))+1);
            end
            empty_image_inds = reshape(empty_image_inds, empty_image_per_batch, []);
            
            % combine sample for each ims_per_batch
            empty_image_inds = reshape(empty_image_inds, empty_image_per_batch, []);
            nonempty_image_inds = reshape(nonempty_image_inds, nonempty_image_per_batch, []);
            
            shuffled_inds = [nonempty_image_inds; empty_image_inds];
            shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));
            
            shuffled_inds = num2cell(shuffled_inds, 1);
        end
    end
    
    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end
