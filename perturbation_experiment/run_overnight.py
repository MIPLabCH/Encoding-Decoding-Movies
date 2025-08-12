import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from dataset import *
from Video_dataset.models_new_2 import *
from Video_dataset.visualisation_new_2 import *



dataset_ID = 6661 # ID of a specific dataset. 6661 refer to preprocessed data with a mask of shape (4609,). 6660 refers to preprocessed data with a mask of shape (15364,)
mask_size = 4609 # number of voxels in the preprocessed fMRI data. either 4609 or 15364
trainset, valset, testset = get_dataset(dataset_ID, mask_size) # data are loaded into dictionaries

print_dict_tree(trainset)



### model training ###

# Training parameters
train_label = trainset['videos']
train_input = trainset['fMRIs']
model = Decoder(mask_size)
num_epochs = 230
lr = 1e-4
#encoder_weight = 0.5
criterion = D_Loss()
#(encoder_weight = encoder_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch_size = 16
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
save_model_as = 'decoder_' + str(mask_size) + '_' + str(num_epochs)
pretrained_decoder = None
start_epoch = 1
start_loss = None
model_to_train = 'decoder'
display_plots = True
save_plots = True
model_name = save_model_as

# Training loop
model, history = train_model(train_input, train_label, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder, start_epoch, start_loss, model_to_train, display_plots, save_plots, model_name)
print_dict_tree(history)

