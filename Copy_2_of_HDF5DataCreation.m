%% WRITING TO HDF5

%%
clc; clear;

%% AD & NC DATA SET

load AD_64x45x18432matlab.mat
load NC_64x45x38144matlab.mat

clear filenames full_name i idx image_folder img imgNII name nn outVolume total_images

[~,~,idx_AD] = size(AD);
[~,~,idx_NC] = size(NC);

% label creation and 4D for data
label_AD = zeros(idx_AD,1);
AD4D = permute(AD, [1 2 4 3]);
label_NC = zeros(idx_NC,1);
NC4D = permute(NC, [1 2 4 3]);

% training testing data creation
indices = randperm(idx_AD);
train_idx = int32(80*idx_AD/100); % 80% training data
train_indices = indices(1:train_idx);
test_indices = indices(train_idx+1:end);

% AD contains the AD MRI images & label_AD contain labels for AD classData,
% we use our random indices to select a subset
% for training and testing
train_data_AD = AD4D(:,:,:,train_indices);
test_data_AD = AD4D(:,:,:,test_indices);

train_labels_AD = label_AD(train_indices);
test_labels_AD = label_AD(test_indices);

% the data is ordered, we want to randomly select 100 points for train, 50
% for test. This part just generates a random list of array indices.
indices = randperm(idx_NC);
train_idx = int32(80*idx_NC/100); % 80% training data
train_indices = indices(1:train_idx);
test_indices = indices(train_idx+1:end);

% NC contains the NC MRI images & label_NC contain labels for NC classData,
% we use our random indices to select a subset
% for training and testing
train_data_NC = NC4D(:,:,:,train_indices);
test_data_NC = NC4D(:,:,:,test_indices);

train_labels_NC = label_NC(train_indices);
test_labels_NC = label_NC(test_indices);


clear AD NC AD4D NC4D idx_AD idx_NC indices label_AD label_NC train_idx

%% shuffl data 

data_training = cat(4, train_data_AD, train_data_NC);
label_training = cat(1, train_labels_AD, train_labels_NC);
[~,~,~,idx_data_train] = size(data_training);
clear train_data_AD train_data_NC train_labels_AD train_labels_NC

indices = randperm(idx_data_train);
shuffled_train_Data = data_training(:,:,:,indices(1,:));
shuffled_train_Label = label_training(indices(1,:),1);
clear data_training label_training indices 

data_testing = cat(4, test_data_AD, test_data_NC);
label_testing = cat(1, test_labels_AD, test_labels_NC);
[~,~,~,idx_data_test] = size(data_testing);
clear test_data_AD test_data_NC test_labels_AD test_labels_NC

indices = randperm(idx_data_test);
shuffled_test_Data = data_testing(:,:,:,indices(1,:));
shuffled_test_Label = label_testing(indices(1,:),1);
clear data_testing label_testing indices

%%

hdf5write('train.hdf5', '/data', shuffled_train_Data );
hdf5write('train.hdf5', '/label', single( permute(reshape(shuffled_train_Label,[idx_data_train, 1, 1, 1]), [4:-1:1] ) ), 'WriteMode', 'append' );


hdf5write('test.hdf5', '/data', shuffled_test_Data );
hdf5write('test.hdf5', '/label', single( permute(reshape(shuffled_test_Label,[idx_data_test, 1, 1, 1]), [4:-1:1] ) ), 'WriteMode', 'append' );

clear idx_data_train idx_data_test test_indices train_indices


% NOTE: In net definition prototxt, use list.txt as input to HDF5_DATA as: 
% layer {
%   name: "data"
%   type: "HDF5Data"
%   top: "data"
%   top: "labelvec"
%   hdf5_data_param {
%     source: "/path/to/list.txt"
%     batch_size: 64
%   }
% }
