function reset_caffe(conf)

    evalc('caffe.reset_all();');
    caffe.set_device(conf.gpu_id-1);
    caffe.set_random_seed(conf.rng_seed);
    caffe.set_mode_gpu();
end
