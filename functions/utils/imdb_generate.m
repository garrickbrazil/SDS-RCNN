function imdb = imdb_generate(root_dir, image_set, flip, cache_dir, dataset)
% imdb = imdb_generate(root_dir, image_set, flip)
%   Package the image annotations into the imdb. 
%
%   Inspired by Ross Girshick's imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------


mkdir_if_missing(cache_dir);

if flip
    cache_file = [cache_dir '/imdb_' dataset '_' image_set '_flip'];
else
    cache_file = [cache_dir '/imdb_' dataset '_' image_set];
end
try
  load(cache_file);
  fprintf('Preloaded imdb %s.. ', image_set);
catch

  fprintf('Computing imdb %s.. ', image_set);
  imdb.name = image_set;
  imdb.extension = '.jpg';
  imdb.image_dir = fullfile(root_dir, image_set, 'images');
  
  imgs = dir(fullfile(imdb.image_dir, ['*' imdb.extension]));

  retain_idx = arrayfun(@(x) isempty(findstr(x.name, 'flip')), imgs);
  imgs = imgs(retain_idx);
  
  imdb.image_ids = cell(length(imgs), 1); 
  
  
  if flip
      for i = 1:length(imgs)
          imdb.image_ids{i*2-1} = imgs(i).name(1:end-4);
          imdb.image_ids{i*2} = [imgs(i).name(1:end-4) '_flip'];
          if ~exist(fullfile(imdb.image_dir, [imgs(i).name(1:end-4) '_flip' imdb.extension]), 'file')  
              im = imread(fullfile(imdb.image_dir, imgs(i).name));
              imwrite(fliplr(im), fullfile(imdb.image_dir, [imgs(i).name(1:end-4) '_flip' imdb.extension]));
          end
      end
  else
      for i = 1:length(imgs)
          imdb.image_ids{i} = imgs(i).name(1:end-4);
      end
  end
  

  imdb.image_at = @(i) ...
      fullfile(imdb.image_dir, [imdb.image_ids{i} imdb.extension]);

  for i = 1:length(imdb.image_ids)
    imageinfo = imfinfo(imdb.image_at(i));
    imdb.sizes(i, :) = [imageinfo.Height imageinfo.Width]; % the size is fix for caltech
  end
  
  imdb.roidb_func = @roidb_generic;
  imdb.num_classes = 1;
  imdb.classes{1} = 'pedestrian';

  save(cache_file, 'imdb', '-v7.3');
end
fprintf('done\n');
end
