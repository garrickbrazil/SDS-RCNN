function model = set_input_data(model, net_inputs, reshape)

    % not reshaping is faster
    if ~exist('reshape', 'var'),  reshape = true;  end

    is_solver = strcmp(class(model), 'caffe.Solver');
    
    if reshape,
        model = reshape_input_data(model, net_inputs);
    end
    
    % set input data
    for i=1:length(net_inputs)
        
        blob = net_inputs{i};
        if is_solver, model.net.blobs(model.net.inputs{i}).set_data(blob);
        else model.blobs(model.inputs{i}).set_data(blob); end
    end
    

end
