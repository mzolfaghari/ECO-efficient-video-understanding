

%----------- net information -----------

save_result_path='.../results/';


ind_split=1;
N_sample=16;


pattern_list={'img_%05d.jpg','%c_%05d.jpg'};
mean_pixel_rgb=[104, 117, 123];
mean_pixel_flow=128;


%======================== test list ====================

list_test_path='/ucf101_RGB_split1_test_demo.txt';

model_file='../ECOfull_ucf101.caffemodel';

model_def_file='/deploy_ECOfull_16.prototxt';
batch_size=80;%
 


if isempty(pattern_list)
    error('Pattern list is empty!!')
end

%---------- dataset information -------------

str_temp=strsplit(model_file,'/');
save_path=sprintf('%s/%s/%s/%s/',save_result_path,cell2mat(str_temp(end-4)),cell2mat(str_temp(end-3)),cell2mat(str_temp(end-2)));
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%----------------- create net info - matfile --------------
clear net_info;
netINFO_mat_path=[save_path,'NET_INFO.mat'];

net_info(1).deploy=model_def_file;
net_info(1).snapshot=model_file;
net_info(1).rgb_mean=mean_pixel_rgb;
net_info(1).flow_mean=mean_pixel_flow;
net_info(1).save_path=save_path;
net_info(1).n_sample=N_sample;
net_info(1).batch_size=batch_size;

net_info(1).pattern=pattern_list{1};

net_info(1).test_list=list_test_path;

save(netINFO_mat_path,'net_info');


%-------------------------------------------------
[list_VidPath,list_startFrame,list_labels]=textread(list_test_path,'%s %d %d');


flag_aug=1;%1 for all fixed crops, 0 for center cropping
N_all=length(list_VidPath);

idxRNG_start=1;
idxRNG_end=N_all;
    
calc_predict(idxRNG_start,idxRNG_end,netINFO_mat_path,ik,specificNumber);


