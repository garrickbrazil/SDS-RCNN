function print_weights(rpn_net)

    for i=1:length(rpn_net.layer_names)
        layername = rpn_net.layer_names{i};
        layer = rpn_net.layers(layername);
        if length(layer.params) > 0
            tmp = (layer.params(1).get_data()); tmp = tmp(:); m = mean(tmp(:));
            fprintf('%s %.20g\n', layername, m);
        end
    end
end
