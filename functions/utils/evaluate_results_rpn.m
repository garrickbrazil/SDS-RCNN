function [mr, recall] = evaluate_results_rpn(conf, net, output_dir, test_dir, test_db)

    if (exist(output_dir, 'dir')), rmdir(output_dir, 's'); end
    
    imlist = dir([test_dir '/images/*.jpg']);
    prevpath = '';
    
    tic;
    
    for imind=1:length(imlist)
        
        %imobj = imlist{imind};
        im = imread([test_dir '/images/' imlist(imind).name]);
        
        %[aboxes, classes] = im_detect(conf, net, imobj.impath, means_obj);
        [pred_boxes, scores] = proposal_im_detect(conf, net, im);
        aboxes = [pred_boxes scores];
        
        [aboxes, valid] = nms_filter(aboxes, conf.nms_per_nms_topN, conf.nms_overlap_thres, conf.nms_after_nms_topN , true);
        
        [~, id] = fileparts(imlist(imind).name);
        
        reg = regexp(id, '(set\d\d)_(V\d\d\d)_I(\d\d\d\d\d)', 'tokens');
        setname = reg{1}{1};
        vname = reg{1}{2};
        iname = reg{1}{3};
        inum = str2num(iname) + 1;
        
        curpath = [output_dir '/' setname '/' vname '.txt'];
        
        if ~strcmp(curpath, prevpath)
            
            if ~isempty(prevpath), fclose(fid); end
            
            mkdir_if_missing([output_dir '/' setname]);
            fid=fopen(curpath, 'a');
            prevpath = curpath;
            
        end
        
        for boxind=1:size(aboxes,1)
            
            x1 = (aboxes(boxind, 1));
            y1 = (aboxes(boxind, 2));
            x2 = (aboxes(boxind, 3));
            y2 = (aboxes(boxind, 4));
            score = aboxes(boxind, 5);
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            if score >= 0.001
                fprintf(fid, '%d,%.3f,%.3f,%.3f,%.3f,%.3f\n', [inum x1 y1 w h score]);    
            end
        end
        
        dt = toc/imind;
        
    end
    
    fclose(fid);

    [mr, ~, recall] = evaluate_result_dir({output_dir}, test_db, conf.test_min_h);
    
end