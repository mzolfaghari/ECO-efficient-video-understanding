function  ClipPrediction_cluster(idxRNG_start,idxRNG_end,netINFO_path,ik,specificNumber)

netINFO_mat=load(netINFO_path);
netINFO_mat=netINFO_mat.net_info;
nAug=10;

if length(netINFO_mat)~=1
    error('This code just works for 1-Stream networks');
end

model_def_file=netINFO_mat(1).deploy;
model_file=netINFO_mat(1).snapshot;
batch_size=netINFO_mat(1).batch_size;
save_path=netINFO_mat(1).save_path;
mean_pixel_rgb=netINFO_mat(1).rgb_mean;
mean_pixel_flow=netINFO_mat(1).flow_mean;
num_frame=netINFO_mat(1).n_sample;
codePath=netINFO_mat(1).codePath;
crop_size=224;

imgSizOrg(1)=240;
imgSizOrg(2)=320;

net_info(1).pattern=netINFO_mat(1).pattern;

for ii=1:length(netINFO_mat)
    net_info(ii).test_list=netINFO_mat(ii).test_list;
    net_info(ii).pattern=netINFO_mat(ii).pattern;
    
end

%you must compile your caffe with matlab support, then add caffe path in the below line:
cd ../caffe_3d/

addpath ./matlab/
gpu_id = 0;

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');


for ii=1:length(netINFO_mat)
    [list_path1,~,list_labels1]=textread(net_info(ii).test_list,'%s %d %d');
    net_data(ii).test_list=list_path1;
end
[list_path2,~,list_labels2]=textread(net_info(1).test_list,'%s %d %d');

addpath(codePath);

nAug=10;
jj=1;
for ii=idxRNG_start:idxRNG_end
    ii
    
    stream2_pathSvTmp=cell2mat(list_path2(ii));
    
    groundTruth_label=list_labels1(ii);
    
    stream2_pathSv=stream2_pathSvTmp;
    str_sp_temp=strsplit(stream2_pathSv,'/');
    str_sp_tempOr=str_sp_temp(~cellfun('isempty',str_sp_temp));
    save_folder=sprintf('%s/',save_path);
    
    if ~exist(save_folder,'dir')
        mkdir(save_folder);
    end
    
    save_result_path=sprintf('%s/all_result_%d_snap%d.mat',save_folder,ik,specificNumber);
    if ~exist(save_result_path)
        prediction_rgb = [];
        
            %===================== stream ===================
            len_video=calc_start_frames(stream2_pathSv);
            
            %==================================================================
            stream1_pathTmp=cell2mat(net_data(1).test_list(ii));
            
            duration = len_video;
            step_frm = floor((duration)/(num_frame));
            
            video_result=[];
            

            
            dataTmpSt1 = zeros(imgSizOrg(1),imgSizOrg(2),3,num_frame,'single');
            data_flipTmpSt1 = zeros(imgSizOrg(1),imgSizOrg(2),3,num_frame,'single');
            rgb_path=cell2mat(net_data(1).test_list(ii));
            
            % selection
            i =1;
            for idx_img = 1:num_frame
                
                img_org = (imread(sprintf(strcat('%s/',net_info(1).pattern), rgb_path, (idx_img-1)*step_frm+1)));
                img=single(imresize(img_org,[imgSizOrg(1),imgSizOrg(2)]));
                dataTmpSt1(:,:,:,i) = img;
                data_flipTmpSt1(:,:,:,i) = img(:,end:-1:1,:);
                i = i + 1;
                
            end
            
            
            %===========================================================
            
            % crop
            rgb_1 = dataTmpSt1(1:224,1:224,:,:);
            rgb_2 = dataTmpSt1(1:224,end-223:end,:,:);
            rgb_3 = dataTmpSt1(16:16+223,59:282,:,:);
            rgb_4 = dataTmpSt1(end-223:end,1:224,:,:);
            rgb_5 = dataTmpSt1(end-223:end,end-223:end,:,:);
            rgb_f_1 = data_flipTmpSt1(1:224,1:224,:,:);
            rgb_f_2 = data_flipTmpSt1(1:224,end-223:end,:,:);
            rgb_f_3 = data_flipTmpSt1(16:16+223,59:282,:,:);
            rgb_f_4 = data_flipTmpSt1(end-223:end,1:224,:,:);
            rgb_f_5 = data_flipTmpSt1(end-223:end,end-223:end,:,:);
            
            rgb = cat(4,rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_f_1,rgb_f_2,rgb_f_3,rgb_f_4,rgb_f_5);
            
            % substract mean and permute
            IMAGE_MEAN = reshape(mean_pixel_rgb,[1,1,3]);
            rgb = bsxfun(@minus,rgb(:,:,[3,2,1],:),IMAGE_MEAN);
            rgb = permute(rgb,[2,1,3,4]);
            
            % predict
            prediction = zeros(51,floor(size(rgb,4)/num_frame));
            num_batches = ceil(size(rgb,4)/batch_size);
            rgbs = zeros(224,224,3,batch_size,'single');
            
            for bb = 1:num_batches
                range = 1 + batch_size*(bb-1): min(size(rgb,4),batch_size*bb);
                range_save = 1 + floor(batch_size/num_frame)*(bb-1): min(floor(size(rgb,4)/num_frame),floor(batch_size/num_frame)*bb);
                
                rgbs(:,:,:,mod(range-1,batch_size)+1) = rgb(:,:,:,range);
                out_put_2stream = net.forward({rgbs});
                out_put_rgb = squeeze(out_put_2stream{1});
                
                prediction(:,range_save) = out_put_rgb(:,mod(range_save-1,floor(batch_size/num_frame))+1);
                
            end

            %
            prediction_rgb = prediction;
            
        
    end
 
    
    score_mean_rgb_NOZ=mean(prediction_rgb');

    [meanVal,meanLabel_rgb]=max(score_mean_rgb_NOZ);
    %================ gather results ===========
    
    mat_score_mean_rgb_noz(jj,:)=score_mean_rgb_NOZ;    
    mat_mean_labels_rgb(jj)=meanLabel_rgb(1);
    
    mat_gt_labels(jj)=list_labels1(ii)+1;
    jj = jj +1;
    
end

all_result.score_mean_rgb_noz=mat_score_mean_rgb_noz;
all_result.labels_mean_rgb=mat_mean_labels_rgb;

all_result.acc_labels_mean_rgb=length(find(mat_mean_labels_rgb==mat_gt_labels))/length(mat_gt_labels);

all_result.mat_gt_labels=mat_gt_labels;

save(save_result_path,'all_result');

caffe.reset_all();

end

function len_video=calc_start_frames(video_name)
%calculate start of frames
framesList_temp = dir([video_name,'/img*.jpg']);
framelist_temp={framesList_temp(:).name};
framelist = setdiff(framelist_temp,{'.','..','.DS_Store','.AppleDouble','.compress_folders.sh.swo'});
len_video=floor(length(framelist)/1);
list_starts=[];


end






