function cur_results = check_loss_rpn(conf, caffe_solver, cur_results)

    % loss values
    for lossind=1:length(conf.loss_layers)
        
        loss_layer = conf.loss_layers{lossind};
        
        if ~isfield(cur_results, loss_layer), cur_results.(loss_layer) = []; end
        
        lossval = caffe_solver.net.blobs(loss_layer).get_data();
        
        if lossval > -1
            
            % artificial boost for bbox regression
            if strcmp(loss_layer, 'loss_bbox'), lossval = lossval*1; end
            
            cur_results.(loss_layer) = [cur_results.(loss_layer) lossval];
        end
        
    end
    
    % accuracy
    labels = caffe_solver.net.blobs('labels_reshape').get_data();
    pred = caffe_solver.net.blobs('proposal_cls_score_reshape').get_data();

    [~, pred] = max(pred, [], 3);
    labels = labels(:); 
    pred = pred(:)-1;
    
    fg_acc = sum(pred(labels>0)==labels(labels>0))/sum(labels>0);
    bg_acc = sum(pred(labels==0)==labels(labels==0))/sum(labels==0);
    acc    = sum(pred==labels)/length(labels);
    
    if ~isfield(cur_results, 'fg_acc'), cur_results.fg_acc = []; end
    if ~isfield(cur_results, 'bg_acc'), cur_results.bg_acc = []; end
    if ~isfield(cur_results, 'acc'),    cur_results.acc    = []; end
    
    if sum(labels>0)  > 0, cur_results.fg_acc = [cur_results.fg_acc fg_acc]; end
    if sum(labels==0) > 0, cur_results.bg_acc = [cur_results.bg_acc bg_acc]; end
    cur_results.acc    = [cur_results.acc acc];
    
end
