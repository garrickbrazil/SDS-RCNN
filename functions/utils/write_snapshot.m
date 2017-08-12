function model_path = write_snapshot(conf, caffe_solver, file_name)
    
    % merge bbox_means, bbox_stds into the model
    anchor_size = size(conf.anchors, 1);
    bbox_pred_layer_name = {'proposal_bbox_pred'};
    
    bbox_stds_flatten = repmat(reshape(conf.bbox_stds', [], 1), anchor_size, 1);
    bbox_means_flatten = repmat(reshape(conf.bbox_means', [], 1), anchor_size, 1);
    weights_back = cell(length(bbox_pred_layer_name), 1);
    biase_back = cell(length(bbox_pred_layer_name), 1);
    
    for i=1:length(bbox_pred_layer_name)
        
        weights = caffe_solver.net.params(bbox_pred_layer_name{i}, 1).get_data();
        biase = caffe_solver.net.params(bbox_pred_layer_name{i}, 2).get_data();
        weights_back{i} = weights;
        biase_back{i} = biase;

        weights = ...
            bsxfun(@times, weights, permute(bbox_stds_flatten, [2, 3, 4, 1])); % weights = weights * stds; 
        biase = ...
            biase .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;

        caffe_solver.net.set_params_data(bbox_pred_layer_name{i}, 1, weights);
        caffe_solver.net.set_params_data(bbox_pred_layer_name{i}, 2, biase);

    end
    
    model_path = fullfile(conf.weights_dir, file_name);
    caffe_solver.net.save(model_path);
    
    % restore net to original state
    for i=1:length(bbox_pred_layer_name)
        caffe_solver.net.set_params_data(bbox_pred_layer_name{i}, 1, weights_back{i});
        caffe_solver.net.set_params_data(bbox_pred_layer_name{i}, 2, biase_back{i});
    end
    
end
