function train_all(rpn_config, rcnn_config, gpu_id)
% train_all
% convienence script to do the following:
%  1. train rpn
%  2. extract proposals
%  3. train rcnn

    if ~exist('rpn_config',  'var'),  error('Please provide rpn config');   end
    if ~exist('rcnn_config', 'var'),  error('Please provide rcnn config');  end
    if ~exist('gpu_id',      'var'),  gpu_id =  1;                          end
    
    % ================================================
    % train rpn
    % ================================================ 
    out_jargin = train_rpn(rpn_config, gpu_id);
    
    
    % ================================================
    % train rcnn
    % ================================================ 
    train_rcnn(rcnn_config, out_jargin.output_dir, ...
        out_jargin.final_model_path, gpu_id)
    
end
