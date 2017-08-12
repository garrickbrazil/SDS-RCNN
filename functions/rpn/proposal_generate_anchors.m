function anchors = proposal_generate_anchors(conf)
% anchors = proposal_generate_anchors(conf)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2015, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    anchor_cache_file           = [conf.output_dir '/anchors'];
    try
        ld                      = load(anchor_cache_file);
        anchors                 = ld.anchors;
    catch
        base_anchor             = [1, 1, conf.base_anchor_size, conf.base_anchor_size];
        ratio_anchors           = ratio_jitter(base_anchor, 1/conf.anchor_ratios);
        anchors                 = cellfun(@(x) scale_jitter(x, conf.anchor_scales), num2cell(ratio_anchors, 2), 'UniformOutput', false);
        anchors                 = cat(1, anchors{:});
        save(anchor_cache_file, 'anchors');
    end
    
end

function anchors = ratio_jitter(anchor, ratios)
    ratios = ratios(:);
    
    w = anchor(3) - anchor(1) + 1;
    h = anchor(4) - anchor(2) + 1;
    x_ctr = anchor(1) + (w - 1) / 2;
    y_ctr = anchor(2) + (h - 1) / 2;
    size = w * h;
    
    size_ratios = size ./ ratios;
    ws = round(sqrt(size_ratios));
    hs = round(ws .* ratios);
    
    anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end

function anchors = scale_jitter(anchor, scales)
    scales = scales(:);

    w = anchor(3) - anchor(1) + 1;
    h = anchor(4) - anchor(2) + 1;
    x_ctr = anchor(1) + (w - 1) / 2;
    y_ctr = anchor(2) + (h - 1) / 2;

    ws = w * scales;
    hs = h * scales;
    
    anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end

