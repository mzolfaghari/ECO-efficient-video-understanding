
% Create list of videos

%Paths
path_save_list='/kinetics/data_list/';
path_DB_rgb='/datasets/kinetics/train/db_frames//';

%Parameters

path_trainList_rgb=fullfile(path_save_list,'kinetics_rgb_train.txt');

fileIDTrain_rgb = fopen(path_trainList_rgb,'w');


action_map_path='class_ind_map_kinetics.mat';

action_map=load(action_map_path);
action_map=action_map.class_ind;


action_list_vid = clean_dir(path_DB_rgb);

for i_activity=1:length(action_list_vid)
    i_activity
    vid_list_path=fullfile(path_DB_rgb,action_list_vid{i_activity});
    video_list_act = clean_dir(vid_list_path);
    action_label=find(strcmp({action_map.new_name}, action_list_vid{i_activity})==1)-1;
    
    for i_video=1:length(video_list_act)
        display(['activity: ',num2str(i_activity),':',action_list_vid{i_activity},'-video:',video_list_act{i_video}])
        path_data_rgb=fullfile(path_DB_rgb,action_list_vid{i_activity},video_list_act{i_video},'/');
        
        len_video=calc_video_len(path_data_rgb);
        if (~isempty(len_video))&&(len_video>5)
            
            fprintf(fileIDTrain_rgb,'%s %d %d\r\n',path_data_rgb,len_video,action_label);
            
            
            
        end
    end
end

fclose(fileIDTrain_rgb);



function files = clean_dir(base)
%clean_dir just runs dir and eliminates files in a folder
files = dir(base);
files_tmp = {};
for i = 1:length(files)
    if strncmpi(files(i).name, '.',1) == 0
        files_tmp{length(files_tmp)+1} = files(i).name;
    end
end
files = files_tmp;
end

function len_video=calc_video_len(video_name)
%calculate the length of videos
framesList_temp = dir([video_name,'/*.jpg']);
framelist_temp={framesList_temp(:).name};
framelist = setdiff(framelist_temp,{'.','..','.DS_Store','.AppleDouble','.compress_folders.sh.swo'});
len_video=length(framelist);
end







