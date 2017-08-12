function model = reshape_input_data(model, net_inputs, hard)

    % not reshaping is faster
    if ~exist('hard', 'var'),  hard = true;  end
    
    is_solver = strcmp(class(model), 'caffe.Solver');

    % reshape
    for i=1:length(net_inputs)
        
        blob = net_inputs{i};
        
        if hard
            dims = length(size(blob));
            reshape_size = ones(1,4);
            reshape_size(1:dims) = size(blob);
        else
            reshape_size = size(blob);
        end
        
        if is_solver, model.net.blobs(model.net.inputs{i}).reshape(reshape_size);
        else model.blobs(model.inputs{i}).reshape(reshape_size); end
    end
    
    if is_solver, model.net.reshape();
    else model.reshape(); end
    
end
