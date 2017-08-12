function out_jargin = train_rpn(config_name, gpu_id, solverstate)

    % ================================================
    % basic configuration
    % ================================================ 
    
    % defaults
    if ~exist('config_name', 'var'),  error('Please provide config');  end
    if ~exist('gpu_id',      'var'),  gpu_id       =  1;               end
    if ~exist('solverstate', 'var'),  solverstate  =  '';              end
    
    rpn_conf = Config.rpn.(config_name);
    
    rpn_conf.stage        =  'rpn';
    rpn_conf.config_name  =  config_name;
    
    % store config
    output_dir = [pwd '/output/' rpn_conf.stage '/' rpn_conf.config_name];
    mkdir_if_missing(output_dir);
    save([output_dir '/rpn_conf.mat'], 'rpn_conf');
    
    % extra misc
    rpn_conf.show_plot    = ~(usejava('jvm') && ~feature('ShowFigureWindows')) && 1;
    rpn_conf.gpu_id       = gpu_id;
    rpn_conf.solverstate  = solverstate;
    
    % extra paths
    rpn_conf.base_dir     = pwd;
    rpn_conf.output_dir   = output_dir;
    rpn_conf.model_dir    = [rpn_conf.base_dir '/models/' rpn_conf.stage '/' rpn_conf.model];
    rpn_conf.init_weights = [rpn_conf.base_dir '/pretrained/' rpn_conf.pretrained];
    rpn_conf.train_dir    = [rpn_conf.base_dir '/datasets/' rpn_conf.dataset_train '/train'];
    rpn_conf.test_dir     = [rpn_conf.base_dir '/datasets/' rpn_conf.dataset_test  '/test'];
    rpn_conf.val_dir      = [rpn_conf.base_dir '/datasets/' rpn_conf.dataset_val   '/val'];
    rpn_conf.weights_dir  = [rpn_conf.output_dir '/weights'];
    rpn_conf.solver_path  = [rpn_conf.output_dir '/solver.prototxt'];
    rpn_conf.train_path   = [rpn_conf.output_dir '/train.prototxt'];
    rpn_conf.test_path    = [rpn_conf.output_dir '/test.prototxt'];
    rpn_conf.cache_dir    = [rpn_conf.base_dir   '/datasets/cache'];
    rpn_conf.log_dir      = [rpn_conf.output_dir '/log'];
    
    % ================================================
    % setup
    % ================================================ 

    mkdir_if_missing(rpn_conf.weights_dir);
    mkdir_if_missing(rpn_conf.log_dir);
    
    copyfile([rpn_conf.model_dir '/train.prototxt'], rpn_conf.train_path);
    copyfile([rpn_conf.model_dir '/test.prototxt' ], rpn_conf.test_path);
    
    % imdb and roidb
    imdb_train    = imdb_generate(['datasets/' rpn_conf.dataset_train], 'train', false, rpn_conf.cache_dir, rpn_conf.dataset_train);
    roidb_train   = roidb_generate(imdb_train, false, rpn_conf.cache_dir, rpn_conf.dataset_train, rpn_conf.min_gt_height);
    
    % anchors
    rpn_conf.anchors = proposal_generate_anchors(rpn_conf);
    
    % misc
    write_solver(rpn_conf);
    reset_caffe(rpn_conf);
    rng(rpn_conf.mat_rng_seed);
    warning('off', 'MATLAB:class:DestructorError');

    % solver
    caffe_solver = caffe.Solver(rpn_conf.solver_path);
    caffe_solver.net.copy_from(rpn_conf.init_weights);
    
    if length(rpn_conf.solverstate)
        caffe_solver.restore([rpn_conf.output_dir '/' rpn_conf.solverstate '.solverstate']);
    end
    
    % ================================================
    % precompute regressions for all images
    % ================================================ 
    
    roidb_train_cache_file  =  [rpn_conf.output_dir '/image_roidb_train.mat'];
    bbox_means_cache_file   =  [rpn_conf.output_dir '/bbox_means.mat'];
    bbox_stds_cache_file    =  [rpn_conf.output_dir '/bbox_stds.mat'];
    
    % preload regression targets
    if exist(roidb_train_cache_file, 'file')==2 && exist(bbox_means_cache_file, 'file')==2 && exist(bbox_stds_cache_file, 'file')==2
        fprintf('Preloading regression targets..');
        load(roidb_train_cache_file);
        load(bbox_means_cache_file);
        load(bbox_stds_cache_file);
        fprintf('Done.\n');
    
    % compute regression targets
    else
        fprintf('Preparing regression targets..');
        [image_roidb_train, bbox_means, bbox_stds] = proposal_prepare_image_roidb(rpn_conf, imdb_train, roidb_train);
        save(roidb_train_cache_file, 'image_roidb_train', '-v7.3');
        save(bbox_means_cache_file,  'bbox_means', '-v7.3');
        save(bbox_stds_cache_file,   'bbox_stds', '-v7.3');
        fprintf('Done.\n');
    end
    
    rpn_conf.bbox_means = bbox_means;
    rpn_conf.bbox_stds = bbox_stds;
        
    % ================================================
    % train
    % ================================================  
    
    % training
    batch = [];
    all_results = {};
    cur_results = {};
    val_mr      = [];
    
    rpn_conf.loss_layers = find_loss_layers(caffe_solver);
    rpn_conf.iter        = caffe_solver.iter();
    
    % already trained?
    if exist([rpn_conf.output_dir sprintf('/weights/snap_iter_%d.caffemodel', rpn_conf.max_iter)], 'file')
        rpn_conf.iter = rpn_conf.max_iter+1;
        fprintf('Final model already exists.\n');
        
    else 
       
        close all; clc; tic;
            
        % log
        curtime = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
        diary([rpn_conf.log_dir '/train_' curtime]);
        caffe.init_log([rpn_conf.log_dir '/caffe_' curtime]);

        disp('conf:'); disp(rpn_conf);
        print_weights(caffe_solver.net);
        
    end
    
    while rpn_conf.iter <= rpn_conf.max_iter
        
        % get samples
        [batch, sampleinds] = proposal_generate_batch(batch, image_roidb_train, 1, rpn_conf.fg_image_ratio);        
        net_inputs = proposal_generate_minibatch(rpn_conf, image_roidb_train(sampleinds));
        
        % set input & step
        caffe_solver = reshape_input_data(caffe_solver, net_inputs);
        caffe_solver = set_input_data(caffe_solver, net_inputs);
        caffe_solver.step(1);
        
        % check loss
        cur_results = check_loss_rpn(rpn_conf, caffe_solver, cur_results);
        rpn_conf.iter = caffe_solver.iter();
        
        % -- print stats --
        if mod(rpn_conf.iter, rpn_conf.display_iter)==0
            
            loss_str = '';
            
            for lossind=1:length(rpn_conf.loss_layers)
        
                loss_name = rpn_conf.loss_layers{lossind};
                loss_val = mean(cur_results.(loss_name));
                
                loss_str = [loss_str sprintf('%s %.3g', strrep(loss_name, 'loss_',''), loss_val)];
                if lossind ~= length(rpn_conf.loss_layers), loss_str = [loss_str ', ']; end
                
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
            
            dt = toc/(rpn_conf.display_iter); tic;
            timeleft = max(dt*(rpn_conf.max_iter - rpn_conf.iter),0);
            if timeleft > 3600, timeleft = sprintf('%.1fh', timeleft/3600);
            elseif timeleft > 60, timeleft = sprintf('%.1fm', timeleft/60);
            else timeleft = sprintf('%.1fs', timeleft); end
            
            fprintf('Iter %d, acc %.2f, fg_acc %.2f, bg_acc %.2f, loss (%s), dt %.2f, eta %s\n', ...
                rpn_conf.iter, all_results.acc(end), all_results.fg_acc(end), ...
                all_results.bg_acc(end), loss_str, dt, timeleft);
            
            update_diary();
            
        end
        
        % -- test net --
        if mod(rpn_conf.iter, rpn_conf.snapshot_iter)==0
            
            % net
            snapped_file = write_snapshot(rpn_conf, caffe_solver, sprintf('snap_iter_%d.caffemodel', rpn_conf.iter));
            results_dir = [rpn_conf.output_dir '/results/val_iter_' num2str(round(rpn_conf.iter/1000)) 'k'];
            solverstate_path = [rpn_conf.output_dir '/weights/snap_iter_' num2str(rpn_conf.iter)];
            
            reset_caffe(rpn_conf);
            
            % val set
            fprintf('Processing val for iter %d..', rpn_conf.iter);

            % test net
            net = caffe.Net([rpn_conf.model_dir '/test.prototxt'], 'test');
            net.copy_from(snapped_file);
            
            % evaluate
            [mr, recall] = evaluate_results_rpn(rpn_conf, net, results_dir, rpn_conf.val_dir, rpn_conf.val_db);
            fprintf('mr %.4f, recall %.4f\n', mr, recall);
            val_mr(length(val_mr)+1) = mr;
            
            clear net;
            
            reset_caffe(rpn_conf);
            
            % restore solver
            caffe_solver = caffe.get_solver(rpn_conf.solver_path);
            caffe_solver.restore([solverstate_path '.solverstate']);
            
        end
        
        % -- plot graphs --
        if rpn_conf.show_plot && mod(rpn_conf.iter, rpn_conf.display_iter)==0

            x = rpn_conf.display_iter:rpn_conf.display_iter:rpn_conf.iter;

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

            loss_legend = cell(length(rpn_conf.loss_layers),1);
            for lossind=1:length(rpn_conf.loss_layers)

                loss_name = rpn_conf.loss_layers{lossind};
                loss_legend{lossind} = strrep(loss_name, '_', '-');
                plot(x, all_results.(loss_name));
                hold on;
            end
            legend(loss_legend);
            hold off;

            drawnow;

        end
        
    end

    if length(val_mr) < length(rpn_conf.snapshot_iter:rpn_conf.snapshot_iter:rpn_conf.max_iter)
        
        val_mr = [];
        
        for iter=rpn_conf.snapshot_iter:rpn_conf.snapshot_iter:rpn_conf.max_iter
            
            % val set
            fprintf('Processing val for iter %d.. ', iter);
            
            results_dir = [rpn_conf.output_dir '/results/val_iter_' num2str(round(iter/1000)) 'k'];
            mr = evaluate_result_dir({results_dir}, rpn_conf.val_db, rpn_conf.test_min_h);
            
            fprintf('MR=%.4f\n', mr);
            val_mr(length(val_mr)+1) = mr;
            
        end
        
    end
    [minval_val, minind_val]   = min(val_mr);
    best_iter = minind_val*rpn_conf.snapshot_iter;
    
    fprintf('Best val=iter_%d at %.4fMR\n',  best_iter, minval_val);
    
    try
        fprintf('Processing final test for iter %d..', best_iter);
        [mr, recall] = evaluate_results_rpn(rpn_conf, net, results_dir, rpn_conf.test_dir, rpn_conf.test_db);
        fprintf('mr %.4f, recall %.4f\n', mr, recall);

    catch
        
        reset_caffe(rpn_conf);
            
        % net
        results_dir = [rpn_conf.output_dir '/results/test_iter_' num2str(round(best_iter/1000)) 'k'];
        solverstate_path = [rpn_conf.output_dir '/weights/snap_iter_' num2str(best_iter)];

        % test net
        net = caffe.Net([rpn_conf.model_dir '/test.prototxt'], 'test');
        net.copy_from([solverstate_path '.caffemodel']);

        % evaluate
        [mr, recall] = evaluate_results_rpn(rpn_conf, net, results_dir, rpn_conf.test_dir, rpn_conf.test_db);
        fprintf('mr %.4f, recall %.4f\n', mr, recall);

        reset_caffe(rpn_conf);

        % restore solver
        caffe_solver = caffe.get_solver(rpn_conf.solver_path);
        caffe_solver.restore([solverstate_path '.solverstate']); 
                
    end
    
    clear net;
    clear caffe_solver;
    
    out_jargin.final_model_path  =  [rpn_conf.output_dir sprintf('/weights/snap_iter_%d.caffemodel', best_iter)];
    out_jargin.output_dir        =  rpn_conf.output_dir;
    
    fprintf('Finished training rpn for %s.\n', config_name);
    
end
