function [mr_fused, mr_rcnn] = evaluate_results_rcnn(rcnn_conf, net, test_rois, db, results_dir_fused, results_dir_rcnn, mode)


    if (exist(results_dir_fused, 'dir')), rmdir(results_dir_fused, 's'); end
    if (exist(results_dir_rcnn, 'dir')),  rmdir(results_dir_rcnn, 's'); end

    for testind=1:length(test_rois)

        rois = test_rois{testind};

        reg = regexp(rois.image_id, '(set\d\d)_(V\d\d\d)_I(\d\d\d\d\d)', 'tokens');
        setname = reg{1}{1};
        vname = reg{1}{2};
        iname = reg{1}{3};
        inum = str2num(iname) + 1;

        mkdir_if_missing([results_dir_fused '/' setname]);
        mkdir_if_missing([results_dir_rcnn  '/' setname]);

        fid  = fopen([results_dir_fused '/' setname '/' vname '.txt'], 'a');
        fid2 = fopen([results_dir_rcnn  '/' setname '/' vname '.txt'], 'a');

        net_inputs = get_rcnn_batch(rcnn_conf, rois, mode);
        net = reshape_input_data(net, net_inputs);
        net.forward(net_inputs);

        cls_scores_fused = net.blobs('cls_score_sm').get_data();
        cls_scores_fused = cls_scores_fused(end,:);
        cls_scores_fused = cls_scores_fused(:);

        cls_scores_rcnn = net.blobs('cls_score2_sm').get_data();
        cls_scores_rcnn = cls_scores_rcnn(end,:);
        cls_scores_rcnn = cls_scores_rcnn(:);

        boxes = rois.boxes(1:rcnn_conf.test_batch_size,:);

        % score 1 (fused)
        aboxes = [boxes, cls_scores_fused];

        for scoreind=1:size(aboxes,1)

            x1 = aboxes(scoreind, 1);
            y1 = aboxes(scoreind, 2);
            x2 = aboxes(scoreind, 3);
            y2 = aboxes(scoreind, 4);
            score = aboxes(scoreind, 5);

            w = x2 - x1;
            h = y2 - y1;

            fprintf(fid, '%d,%.3f,%.3f,%.3f,%.3f,%.6f\n', [inum x1 y1 w h score]);

        end

        % score 2 (rcnn only)
        aboxes = [boxes, cls_scores_rcnn];

        for scoreind=1:size(aboxes,1)

            x1 = aboxes(scoreind, 1);
            y1 = aboxes(scoreind, 2);
            x2 = aboxes(scoreind, 3);
            y2 = aboxes(scoreind, 4);
            score = aboxes(scoreind, 5);

            w = x2 - x1;
            h = y2 - y1;

            fprintf(fid2, '%d,%.3f,%.3f,%.3f,%.3f,%.6f\n', [inum x1 y1 w h score]);

        end

        fclose(fid);
        fclose(fid2);

    end

    mr_fused = evaluate_result_dir({results_dir_fused}, db, rcnn_conf.test_min_h);
    mr_rcnn  = evaluate_result_dir({results_dir_rcnn}, db, rcnn_conf.test_min_h);

end