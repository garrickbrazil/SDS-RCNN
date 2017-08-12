function write_solver(conf)

    solver_txt = fileread([conf.base_dir '/models/solver_template_' lower(conf.solver_type) '.prototxt']);
    
    solver_txt = strrep(solver_txt, '{train_proto}', [conf.model_dir '/train.prototxt']);
    solver_txt = strrep(solver_txt, '{lr}', num2str(conf.lr));
    solver_txt = strrep(solver_txt, '{step_size}', num2str(conf.step_size));
    solver_txt = strrep(solver_txt, '{max_iter}', num2str(conf.max_iter));
    solver_txt = strrep(solver_txt, '{snapshot_iter}', num2str(conf.snapshot_iter));
    solver_txt = strrep(solver_txt, '{output_dir}', conf.output_dir);
    
    fileID = fopen([conf.solver_path],'w');
    fprintf(fileID,'%s',solver_txt);
    fclose(fileID);

end