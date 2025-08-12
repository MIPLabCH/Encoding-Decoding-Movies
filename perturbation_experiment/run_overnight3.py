### necessary imports ###

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from Video_dataset.dataset_new import *
from Video_dataset.models_new_2 import *
from Video_dataset.visualisation_new_2 import *
from perturbation import *


### data loading ###

dataset_ID = 6661 # ID of a specific dataset. 6661 refer to preprocessed data with a mask of shape (4609,). 6660 refers to preprocessed data with a mask of shape (15364,)
mask_size = 4609 # number of voxels in the preprocessed fMRI data. either 4609 or 15364
trainset, valset, testset = get_dataset(dataset_ID, mask_size) # data are loaded into dictionaries


subject_ids = ['01-norm', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
                '13', '14', '15', '16', '17', '19', '20', '22', '23', '24', 
                '25', '27', '28', '29', '31', '32']

# Convert to folder names format
subjects_list = [f'sub-S{subj_id}' for subj_id in subject_ids]

subjects_list.append('average-norm')

# Base directory containing all subject folders
base_dir = 'processed_data'

# List of subject directories (excluding missing ones)
#subjects = [d for d in os.listdir(base_dir) if d.startswith('sub-S') and d not in ['sub-S12', 'sub-S18']]
# Or explicitly list them
# subjects = ['sub-S01', 'sub-S02', 'sub-S03', ...] # excluding sub-S12 and sub-S18

# Path to videos (labels)
videos_path = 'processed_data/videos/videos.npy'

model = Decoder(mask_size)
num_epochs = 350

lr = 1e-4
#encoder_weight = 0.5
criterion = D_Loss()
#(encoder_weight = encoder_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch_size = 16
#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
save_model_as = 'decoder_all_4609_' + str(num_epochs)
#decoder_all meaning its trained on all subjects instead of average
#pretrained_decoder = None
start_epoch = 1
start_loss = None
model_to_train = 'decoder'
display_plots = True
save_plots = True
model_name = save_model_as

# Call the training function (with your model and other parameters)
model, history = train_model3(
    base_dir=base_dir,
    subjects_list=subjects_list,
    videos_path=videos_path,
    model=model,  # Replace with your actual model
    num_epochs=num_epochs,    # Adjust as needed
    lr=lr,          # Adjust as needed
    criterion=criterion,  # Replace with your actual criterion
    optimizer=optimizer,  # Replace with your actual optimizer
    batch_size=batch_size,     # Adjust as needed
    device=device,
    save_model_as=save_model_as,
    model_to_train=model_to_train,  # or whichever part you're training
    display_plots=display_plots,
    save_plots=save_plots,
    model_name='decoder_all_4609_' + str(num_epochs)
)



