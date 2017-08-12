function train_rcnn(config_name, rpn_dir, rpn_weights, gpu_id, solverstate)

    % ================================================
    % basic configuration
    % ================================================ 
    
    % defaults
    if ~exist('config_name', 'var'),  error('Please provide config');  end
    if ~exist('gpu_id',      'var'),  gpu_id       =  1;               end
    if ~exist('solverstate', 'var'),  solverstate  =  '';              end

    rcnn_conf = Config.rcnn.(config_name);
    
    rcnn_conf.stage        =  'rcnn';
    rcnn_conf.config_name  =  config_name;
    
    % store config
    output_dir = [pwd '/output/' rcnn_conf.stage '/' rcnn_conf.config_name];
    mkdir_if_missing(output_dir);
    save([output_dir '/rcnn_conf.mat'], 'rcnn_conf');
    
    % extra misc
    rcnn_conf.show_plot    = ~(usejava('jvm') && ~feature('ShowFigureWindows')) && 0;
    rcnn_conf.gpu_id       = gpu_id;
    rcnn_conf.solverstate  = solverstate;
    rcnn_conf.rpn_weights  = rpn_weights;
    
    % extra paths
    rcnn_conf.base_dir     = pwd;
    rcnn_conf.output_dir   = output_dir;
    rcnn_conf.model_dir    = [rcnn_conf.base_dir '/models/' rcnn_conf.stage '/' rcnn_conf.model];
    rcnn_conf.train_dir    = [rcnn_conf.base_dir '/datasets/' rcnn_conf.dataset_train '/train'];
    rcnn_conf.test_dir     = [rcnn_conf.base_dir '/datasets/' rcnn_conf.dataset_test  '/test'];
    rcnn_conf.val_dir      = [rcnn_conf.base_dir '/datasets/' rcnn_conf.dataset_val   '/val'];
    rcnn_conf.weights_dir  = [rcnn_conf.output_dir '/weights'];
    rcnn_conf.solver_path  = [rcnn_conf.output_dir '/solver.prototxt'];
    rcnn_conf.train_path   = [rcnn_conf.output_dir '/train.prototxt'];
    rcnn_conf.test_path    = [rcnn_conf.output_dir '/test.prototxt'];
    rcnn_conf.cache_dir    = [rcnn_conf.base_dir '/datasets/cache'];
    rcnn_conf.log_dir      = [rcnn_conf.output_dir '/log'];
    
    % ================================================
    % setup
    % ================================================ 

    mkdir_if_missing(rcnn_conf.weights_dir);
    mkdir_if_missing(rcnn_conf.log_dir);
    
    copyfile([rcnn_conf.model_dir '/train.prototxt'], rcnn_conf.train_path);
    copyfile([rcnn_conf.model_dir '/test.prototxt' ], rcnn_conf.test_path);
    copyfile(rpn_weights, [rcnn_conf.output_dir '/final_rpn.caffemodel']);
    
    % imdb and roidb
    imdb_train    =  imdb_generate(['datasets/' rcnn_conf.dataset_train], 'train', false, rcnn_conf.cache_dir, rcnn_conf.dataset_train);
    imdb_test     =  imdb_generate(['datasets/' rcnn_conf.dataset_test ], 'test',  false, rcnn_conf.cache_dir, rcnn_conf.dataset_test );
    imdb_val      =  imdb_generate(['datasets/' rcnn_conf.dataset_val  ], 'val',   false, rcnn_conf.cache_dir, rcnn_conf.dataset_val  );
    roidb_train   =  roidb_generate(imdb_train, false, rcnn_conf.cache_dir, rcnn_conf.dataset_train, rcnn_conf.min_gt_height);
    roidb_test    =  roidb_generate(imdb_test,  false, rcnn_conf.cache_dir, rcnn_conf.dataset_test,  rcnn_conf.min_gt_height);
    roidb_val     =  roidb_generate(imdb_val,   false, rcnn_conf.cache_dir, rcnn_conf.dataset_val,   rcnn_conf.min_gt_height);
    
    % misc
    write_solver(rcnn_conf);
    reset_caffe(rcnn_conf);
    rng(rcnn_conf.mat_rng_seed);
    warning('off', 'MATLAB:class:DestructorError');
    
    % solver
    caffe_solver = caffe.Solver(rcnn_conf.solver_path);
    caffe_solver.net.copy_from(rcnn_conf.rpn_weights);
    
    if length(rcnn_conf.solverstate)
        caffe_solver.restore([rcnn_conf.output_dir '/' rcnn_conf.solverstate '.solverstate']);
    end
    
    % ================================================
    % extract proposals for all images
    % ================================================ 
    
    % rpn test net
    rpn_net = caffe.Net([rpn_dir '/test.prototxt'], 'test');
    rpn_net.copy_from(rpn_weights);
    
    % config
    load([rpn_dir '/rpn_conf.mat']);
    load([rpn_dir '/anchors.mat']);
    load([rpn_dir '/bbox_means.mat']);
    load([rpn_dir '/bbox_stds.mat']);

    rpn_conf.anchors    = anchors;
    rpn_conf.bbox_means = bbox_means;
    rpn_conf.bbox_stds  = bbox_stds;
    
    rois_train_file  =  [rcnn_conf.output_dir '/rois_train.mat'];
    rois_test_file   =  [rcnn_conf.output_dir '/rois_test.mat'];
    rois_val_file    =  [rcnn_conf.output_dir '/rois_val.mat'];
    
    % preload proposals
    if exist(rois_train_file, 'file')==2 
        fprintf('Preloading train proposals.. ');
        load(rois_train_file);
        fprintf('done\n');
    else
        fprintf('Extracting train proposals..\n');
        rois_train = get_top_proposals(rpn_conf, rpn_net, roidb_train, imdb_train);
        save(rois_train_file, 'rois_train');
    end
    if exist(rois_test_file, 'file')==2
        fprintf('Preloading test proposals.. ');
        load(rois_test_file);
        fprintf('done\n');
    else
        fprintf('Extracting test proposals..\n');
        rois_test = get_top_proposals(rpn_conf, rpn_net, roidb_test, imdb_test);
        save(rois_test_file, 'rois_test');
    end
    if exist(rois_val_file, 'file')==2       
        fprintf('Preloading val proposals.. ');
        load(rois_val_file);
        fprintf('done\n');
    else
        fprintf('Extracting val proposals..\n');
        rois_val = get_top_proposals(rpn_conf, rpn_net, roidb_val, imdb_val);
        save(rois_val_file, 'rois_val');
    end
    
    clear rpn_net;
    
    % ================================================
    % train
    % ================================================
    batch       = randperm(length(rois_train));
    all_results = {};
    cur_results = {};
    val_mr      = [];

    rcnn_conf.loss_layers = find_loss_layers(caffe_solver);
    rcnn_conf.iter        = caffe_solver.iter();
    
    % already trained?
    if exist([rcnn_conf.output_dir sprintf('/weights/snap_iter_%d.caffemodel', rcnn_conf.max_iter)], 'file')
        rcnn_conf.iter = rcnn_conf.max_iter+1;
        fprintf('Final model already exists.\n');
        
    else 
       
        close all; clc; tic;
            
        % log
        curtime = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
        diary([rcnn_conf.log_dir '/train_' curtime]);
        caffe.init_log([rcnn_conf.log_dir '/caffe_' curtime]);

        disp('conf:'); disp(rcnn_conf);
        print_weights(caffe_solver.net);
        
    end
    
    while rcnn_conf.iter <= rcnn_conf.max_iter
        
        % get samples
        inds = batch(mod(rcnn_conf.iter,length(rois_train)) + 1);
        net_inputs = get_rcnn_batch(rcnn_conf, rois_train{inds}, 'train');
                
        % set input & step
        caffe_solver = reshape_input_data(caffe_solver, net_inputs);
        caffe_solver = set_input_data(caffe_solver, net_inputs);
        caffe_solver.step(1);
        
        % check loss
        cur_results = check_loss_rcnn(rcnn_conf, caffe_solver, cur_results);
        rcnn_conf.iter = caffe_solver.iter();
        
        % -- print stats --
        if mod(rcnn_conf.iter, rcnn_conf.display_iter)==0
            
            loss_str = '';
            
            for lossind=1:length(rcnn_conf.loss_layers)
        
                loss_name = rcnn_conf.loss_layers{lossind};
                loss_val = mean(cur_results.(loss_name));
                
                loss_str = [loss_str sprintf('%s %.3g', strrep(loss_name, 'loss_',''), loss_val)];
                if lossind ~= length(rcnn_conf.loss_layers), loss_str = [loss_str ', ']; end
                
                if ~isfield(all_results, loss_name), all_results.(loss_name) = []; end
                all_results.(loss_name) = [all_results.(loss_name); loss_val];
                cur_results.(loss_name) = [];
            end
            
            if ~isfield(all_results, 'acc'),    all_results.acc    = []; end
            if ~isfield(all_results, 'fg_acc'), all_results.fg_acc = []; end
            if ~isfield(all_results, 'bg_acc'), all_results.bg_acc = []; end
            
            all_results.acc    = [all_results.acc    mean(cur_results.acc)];
            all_results.fg_acc = [all_results.fg_acc mean(cur_results.fg_acc)];
            all_results.bg_acc = [all_results.bg_acc mean(cur_results.bg_acc)];
            
            cur_results.acc    = [];
            cur_results.fg_acc = [];
            cur_results.bg_acc = [];
            
            dt = toc/(rcnn_conf.display_iter); tic;
            timeleft = max(dt*(rcnn_conf.max_iter - rcnn_conf.iter),0);
            if timeleft > 3600, timeleft = sprintf('%.1fh', timeleft/3600);
            elseif timeleft > 60, timeleft = sprintf('%.1fm', timeleft/60);
            else timeleft = sprintf('%.1fs', timeleft); end
            
            fprintf('Iter %d, acc %.2f, fg_acc %.2f, bg_acc %.2f, loss (%s), dt %.2f, eta %s\n', ...
                rcnn_conf.iter, all_results.acc(end), all_results.fg_acc(end), ...
                all_results.bg_acc(end), loss_str, dt, timeleft);
            
            update_diary();
            
        end
        
        % -- test/val --
        if mod(rcnn_conf.iter, rcnn_conf.snapshot_iter)==0
            
            reset_caffe(rcnn_conf);
            
            % net
            solverstate_path  = [rcnn_conf.output_dir '/weights/snap_iter_' num2str(rcnn_conf.iter)];
            net = caffe.Net([rcnn_conf.model_dir '/test.prototxt'], 'test');
            net.copy_from([solverstate_path '.caffemodel']);
            
            % val set
            fprintf('Processing val for iter %d.. ', rcnn_conf.iter);
            
            results_dir_fused = [rcnn_conf.output_dir '/results/val_iter_' num2str(round(rcnn_conf.iter/1000)) 'k_fuse'];
            results_dir_rcnn  = [rcnn_conf.output_dir '/results/val_iter_' num2str(round(rcnn_conf.iter/1000)) 'k_rcnn'];
            
            [mr_fused, mr_rcnn] = evaluate_results_rcnn(rcnn_conf, net, rois_val, rcnn_conf.val_db, results_dir_fused, results_dir_rcnn, 'val');
            fprintf('MR=%.4f, Non-fused MR=%.4f\n', mr_fused, mr_rcnn);
            val_mr(length(val_mr)+1) = mr_fused;
            
            clear net;
            reset_caffe(rcnn_conf);
            
            % restore solver
            caffe_solver = caffe.get_solver(rcnn_conf.solver_path);
            caffe_solver.restore([solverstate_path '.solverstate']);
            
            update_diary();
            
        end
        
        % -- plot graphs --
        if rcnn_conf.show_plot && mod(rcnn_conf.iter, rcnn_conf.display_iter)==0

            x = rcnn_conf.display_iter:rcnn_conf.display_iter:rcnn_conf.iter;

            % loss plot
            subplot(1,2,1);

            plot(x,all_results.acc);
            hold on;
            plot(x,all_results.fg_acc);
            plot(x,all_results.bg_acc);
            legend('acc', 'fg-acc', 'bg-acc');
            hold off;

            % loss plot
            subplot(1,2,2);

            loss_legend = cell(length(rcnn_conf.loss_layers),1);
            for lossind=1:length(rcnn_conf.loss_layers)

                loss_name = rcnn_conf.loss_layers{lossind};
                loss_legend{lossind} = strrep(loss_name, '_', '-');
                plot(x, all_results.(loss_name));
                hold on;
            end
            legend(loss_legend);
            hold off;

            drawnow;

        end
        
    end
    
    if length(val_mr) < length(rcnn_conf.snapshot_iter:rcnn_conf.snapshot_iter:rcnn_conf.max_iter)
        
        val_mr = [];
        
        for iter=rcnn_conf.snapshot_iter:rcnn_conf.snapshot_iter:rcnn_conf.max_iter
            
            % val set
            fprintf('Processing val for iter %d.. ', iter);
            
            results_dir_fused = [rcnn_conf.output_dir '/results/val_iter_' num2str(round(iter/1000)) 'k_fuse'];
            results_dir_rcnn  = [rcnn_conf.output_dir '/results/val_iter_' num2str(round(iter/1000)) 'k_rcnn'];
            
            mr_fused = evaluate_result_dir({results_dir_fused}, rcnn_conf.val_db, rcnn_conf.test_min_h);
            mr_rcnn  = evaluate_result_dir({results_dir_rcnn},  rcnn_conf.val_db, rcnn_conf.test_min_h);
            
            fprintf('MR=%.4f, Non-fused MR=%.4f\n', mr_fused, mr_rcnn);
            val_mr(length(val_mr)+1) = mr_fused;
            
        end
        
    end
    
    [minval_val, minind_val]   = min(val_mr);
    best_iter = minind_val*rcnn_conf.snapshot_iter;
    
    fprintf('Best val=iter_%d at %.4fMR\n',  best_iter, minval_val);
    
    results_dir_fused = [rcnn_conf.output_dir '/results/test_iter_' num2str(round(best_iter/1000)) 'k_fuse'];
    results_dir_rcnn  = [rcnn_conf.output_dir '/results/test_iter_' num2str(round(best_iter/1000)) 'k_rcnn'];
    
    try
        fprintf('Processing final test for iter %d.. ', best_iter);
        [mr_fused, mr_rcnn] = evaluate_results_rcnn(rcnn_conf, net, rois_test, rcnn_conf.test_db, results_dir_fused, results_dir_rcnn, 'test');
        fprintf('MR=%.4f, Non-fused MR=%.4f\n', mr_fused, mr_rcnn);
    catch
        
        reset_caffe(rcnn_conf);

        % net
        solverstate_path  = [rcnn_conf.output_dir '/weights/snap_iter_' num2str(best_iter)];
        net = caffe.Net([rcnn_conf.model_dir '/test.prototxt'], 'test');
        net.copy_from([solverstate_path '.caffemodel']);

        [mr_fused, mr_rcnn] = evaluate_results_rcnn(rcnn_conf, net, rois_test, rcnn_conf.test_db, results_dir_fused, results_dir_rcnn, 'test');
        fprintf('MR=%.4f, Non-fused MR=%.4f\n', mr_fused, mr_rcnn);

        clear net;

        reset_caffe(rcnn_conf);
    end
    
    copyfile([rcnn_conf.output_dir '/weights/snap_iter_' num2str(best_iter) '.caffemodel'], [rcnn_conf.output_dir '/final_rcnn.caffemodel']);

end
