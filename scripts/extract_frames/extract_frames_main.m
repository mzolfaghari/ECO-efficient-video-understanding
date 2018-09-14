
function extract_frames_main
% Extract frames using ffmpeg
%Mohammadreza Zolfaghari, University of Freiburg, September 2018
%

path1='/dataset/kinetics/train/'; % Source path for Videos
path2='/dataset/kinetics/train_frm/'; % Save path for frames
frm_rate=25;
folderlist = clean_dir(path1);


index_all=length(folderlist);
for i = 1:index_all
    
    if ~exist([path2,folderlist{i}],'dir')
        mkdir([path2,folderlist{i}]);
    end
    filelist = dir([path1,folderlist{i},'/*.mp4']);
%     
    for j = 1:length(filelist)
        if ~exist([path2,'/',folderlist{i},'/',filelist(j).name(1:end-4)],'dir')
            mkdir([path2,'/',folderlist{i},'/',filelist(j).name(1:end-4)]);
        end
        
        iFileAddressStr=[path1,'/',folderlist{i},'/',filelist(j).name];
        outFolderStr = [path2,'/',folderlist{i},'/',filelist(j).name(1:end-4),'/','img'];
        mycommand = sprintf('sh   extract_frames_frmRate.sh %s %d %s',iFileAddressStr,frm_rate,outFolderStr);
        system(mycommand);

    end
    msg = sprintf('****** Processing: %.2f percent done', (i/index_all)*100);
    fprintf(msg);
end





function files = clean_dir(base)
%clean_dir just runs dir and eliminates files in a foldr
files = dir(base);
files_tmp = {};
for i = 1:length(files)
    if strncmpi(files(i).name, '.',1) == 0
        files_tmp{length(files_tmp)+1} = files(i).name;
    end
end
files = files_tmp;
