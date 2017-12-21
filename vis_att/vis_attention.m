% clear; clc;

TEST_NUM = 20; % Number of test 
QUES_ID_START = 200; % Start of testing question id

image_path = '../../VQA/Images/mscoco/test-dev2015/';  % Image file
% image_path = '/Users/dbxiaolu/Pictures/test-dev2015/'    % Image file
att_h5 = '../result/vqa_test-dev2015_vqa_model_6601_test_#207_atts.h5'; % Attention map file
box_h5 = '../../VQA/Features/faster-rcnn_box4_19_test-dev.h5'; % Bounding box file
ques_json = 'test-dev_prepro_questions.json'; % Question file
% { 4195880:{'question': u'Are the dogs tied?', 'image_name': u'COCO_test2015_000000419588.jpg'}, 
% 4195881 {'question': u'Is this a car show?', 'image_name': u'COCO_test2015_000000419588.jpg'}, }


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read related files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question file
if ~exist('jsonData','var')
    addpath('jsonlab-1.2');
    jsonData = loadjson(ques_json);
    raw_que_list = fieldnames(jsonData);    % question id list
end

que_num = length(raw_que_list)  % # of Questions: 60864
que_num = TEST_NUM; % number of examples to show
que_id1 = QUES_ID_START % Start of testing question id

% Attention file
qids_data = hdf5read(att_h5, 'question_id');
att1_data = hdf5read(att_h5, 'att1'); size(att1_data)
att2_data = hdf5read(att_h5, 'att2'); size(att2_data)
att3_data = hdf5read(att_h5, 'att3'); size(att3_data)
att4_data = hdf5read(att_h5, 'att4'); size(att4_data)

% qids_data(1)
% att1_data(:,:,1)
% att2_data(:,:,1)
% att3_data(:,1)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize attention maps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i= que_id1:que_id1+que_num-1
    fprintf('This is the sample %d.\n\n', i)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Obtain question string, image name
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    raw_que_id = raw_que_list{i}
    %raw_que_id = 'x0x34_195880'

    que_id = strcat(raw_que_id(5), raw_que_id(7:end));  % question id
    fprintf('question id is: %s \n', que_id);
    
    que_struct = getfield(jsonData, raw_que_id);
    que_str = que_struct.question;  % question string
    img_str = que_struct.image_name;    % image name
    
    fprintf('question is: %s \n', que_str);
    fprintf('image name is: %s \n\n',  img_str); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % New firgure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx = find(qids_data==str2num(que_id))
    qids_data(idx)
    
    img_path = strcat(image_path, img_str)
    ori_img = imread(img_path); % load image
    h = size(ori_img,1)
    w = size(ori_img,2)

    figure; % new firgure window
    set(gcf, 'position', [0 0 900 500]); %[left bottom width height]´°¿Ú´óÐ¡:900x900
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Visualize attention map for branch 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(1,2,1)
    imshow(ori_img);    % display the original image
    hold on;

	heatmap1 =  (att1_data(:,:,idx)'+att2_data(:,:,idx)')/2;
    heatmap1 = heatmap1/max(heatmap1(:))*255;
    imagesc(imresize(heatmap1,[h,w]),'AlphaData',0.5);
   
    set(gcf,'color','none');
    axis([0 w 0 h]);
    set(gca,'position',[0 0 1 1]);
    saveas(gcf,strcat(que_id, '_1', '.png')) ;
    close;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Visualize attention map for branch 2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(1,2,2)
    imshow(ori_img);    % display the original image
    hold on;
    
    box_data = hdf5read(box_h5, img_str)'; % Box: 19x4 [Xmin,Ymin,Xmax,Ymax]
    att_data =  (att3_data(:,idx)+att4_data(:,idx))/2;  % attention weight: 19
    
    heatmap3 = zeros(w,h);
    
    for box_id = 1:19
        att = att_data(box_id);
        box = box_data(box_id,:);
        
        x_min = floor(box(1));
        if x_min == 0 
            x_min = 1;
        end
        x_max = floor(box(3));
        y_min = floor(box(2));
        if y_min == 0 
            y_min = 1;
        end
        y_max = floor(box(4));
        
        heatmap3(x_min:x_max, y_min:y_max) = heatmap3(x_min:x_max, y_min:y_max) + att;
    end
    
    heatmap3 = heatmap3';
    heatmap3 = heatmap3/max(heatmap3(:))*255;
    imagesc(imresize(heatmap3,[h,w]),'AlphaData',0.5);
    
    set(gcf,'color','none');
    axis([0 w 0 h]);
    set(gca,'position',[0 0 1 1]);
    saveas(gcf,strcat(que_id, '_2', '.png')) ;
    close;

end
