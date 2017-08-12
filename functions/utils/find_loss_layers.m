function loss_layers = find_loss_layers(caffe_solver)

    loss_layers = {};
    
    layers = caffe_solver.net.blob_names;
    
    for lind=1:length(layers)
       layer = layers{lind};
        if strfind(layer,'loss') == 1
            loss_layers{end+1} = layer;
        end
    end
end