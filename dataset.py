# -*- coding: utf-8 -*-

"""
This file contains functions related to data processing, storing, and loading.

Functions:
    load_data
    load_masked_fMRI
    load_processed_videos
    load_durations
    store_data
    store_masked_fMRI
    store_average_fMRIs
    store_processed_videos
    normalize
    process_frame
    split_for_encoder
    get_dataset
"""

from imports import os, np, nib, pd, ThreadPoolExecutor, resize, imageio, time, torch


### DATA LOADING ###


def load_data(fMRIs_path = 'fMRIs', subject = 'average', videos_path = 'processed_videos', video = 'all'):
    """
    Load preprocessed fMRI data videos.
    
    Arguments:
        fMRIs_path (str): Path to the directory containing preprocessed fMRI data.
        subject (str): Specifies the subject to load data for. If 'average', load average fMRI data across multiple subjects. Else, load for a specific subject (ex: 'sub-S01').
        videos_path (str): Path to the directory containing processed videos. 
        video (str): Specifies which videos to load, 'all' loads all videos. If 'all', load all videos. Else, load a specific movie (ex: 'YouAgain').

    Returns:
        tuple: A tuple containing two dictionaries: the fMRI data and video data.
    """ 
    return load_masked_fMRI(fMRIs_path, subject), load_processed_videos(videos_path, video)

def load_masked_fMRI(fMRIs_path = 'fMRIs', subject = 'average'):
    """
    Load masked fMRI data.
    
    Arguments:
        fMRIs_path (str): Path to the directory containing preprocessed fMRI data.
        subject (str): Specifies the subject to load data for. If 'average', load average fMRI data across multiple subjects. Else, load for a specific subject (ex: 'sub-S01').
        
    Returns:
        dict: A dictionary where each key is a video name and its value is corresponding fMRI data.
    """
    datas = {}
    data_path = os.path.join(fMRIs_path, subject)
    if not os.path.exists(data_path):
        print(f'{data_path} does not exist.')
        return None
    # iterate through all the files of the folder
    for file in os.listdir(data_path):
        name = file[:-4]     # -4 to remove the extension '.npy' 
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path) and file[0] != '.':
            data = np.load(file_path, mmap_mode = 'r')
            if subject != 'average':
                onset, duration = load_durations(video = name, subject = subject[4:])
                data = data[:,onset:onset+duration]
                data = np.transpose(data,(1,0)) 
            #else: # you can z-score the data for training
            #    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0) #z-score the average fMRI
            datas[name] = data

    return datas

def load_processed_videos(videos_path = 'processed_videos', video = 'all'):
    """
    Load preprocessed video data.
    
    Arguments:
        videos_path (str): Path to the directory containing processed videos. 
        video (str): Specifies which videos to load, 'all' loads all videos. If 'all', load all videos. Else, load a specific movie (ex: 'YouAgain').

    Returns:
        dict: A dictionary where each key is a movie title and its value is the corresponding video data.
    """
    if not os.path.exists(videos_path):
        print(f'{videos_path} does not exist.')
        return None
    # store the preprocessed videos in the dictionary vids. The keys are the titles of the movies, the elements are the arrays
    videos = {}
    if video == 'all':
        # iterate through all the files of the folder
        for file in os.listdir(videos_path):
            name = file[:-4]     # -4 to remove the extension '.npy' 
            file_path = os.path.join(videos_path, file)
            if os.path.exists(file_path) and file[0] != '.':
                videos[name] = np.load(file_path, mmap_mode = 'r')
                
    # store the preprocessed video in the array vids
    else:
        file_path = os.path.join(videos_path, video+'.npy')
        if os.path.exists(file_path):
            videos[video] = np.load(file_path, mmap_mode = 'r')
        else:
            print(f'{file_path} does not exist.')
            return None

    return videos

def load_durations(video, subject):
    """
    Load the fMRI onset and duration for a specific video and subject from a predefined path.

    Arguments:
        video (str): Name of the video.
        subject (str): Subject identifier.

    Returns:
        tuple: A tuple containing the onset time and video duration.
    """

    # you can change the path of the file containing the video durations
    durations_path = '/media/miplab-nas-shadow/Data2/Michael-GOTO-RCPNAS/ml-students2023/resources/run_onsets.pkl'
    onset = pd.read_pickle(durations_path)[video][subject][0]
    video_duration = load_processed_videos(video=video)[video].shape[0]

    return onset, video_duration

### fMRI MASK ###

def get_schaefer1000_mask(save_as):
    # load complete schaefer mask
    schaefer = 'Schaefer2018_1000Parcels_Kong2022_17Networks_order_FSLMNI152_2mm.nii.gz'
    img_schaefer = nib.load(schaefer)
    data_schaefer = np.asanyarray(img_schaefer.dataobj)

    # select 15364 voxels from the visual areas
    ROI = {'17networks_LH_VisualA_ExStr': [420, 450],
           #'17networks_LH_VisualA_PrC': [451, 452],
           #'17networks_LH_VisualA_SPL': [453, 454],
           #'17networks_LH_VisualB_ExStrInf': [455, 463],
           #'17networks_LH_VisualB_ExStrSup': [464, 477],
           '17networks_LH_VisualB_Striate': [478, 488],
           '17networks_LH_VisualC_ExStr': [489, 496],
           '17networks_LH_VisualC_Striate': [497, 500],
           '17networks_RH_VisualA_ExStr': [910, 940],
           #'17networks_RH_VisualA_PrC': [941, 942],
           #'17networks_RH_VisualA_SPL': [943, 949],
           #'17networks_RH_VisualB_ExStrInf': [950, 959],
           #'17networks_RH_VisualB_ExStrSup': [960, 971],
           #'17networks_RH_VisualB_SPL': [972, 975],
           '17networks_RH_VisualB_Striate': [976, 991],
           '17networks_RH_VisualC_ExStr': [992, 1000]}

    mask = np.zeros((91, 109, 91))
    for roi_name in ROI.keys():
        mask += ((data_schaefer >= ROI[roi_name][0]) & (data_schaefer <= ROI[roi_name][1])).astype(int)

    np.save(save_as, mask)
    print(f"Mask created (shape: {mask.shape})")

def get_snr_mask(percentile = 70):

    movies = {}
    for movie in os.listdir('processed_videos'):
        if movie[:11] != 'ForrestGump' and movie[0] != '.':
            name = movie[:-4]
            print(name)
            movies[name] = [load_masked_fMRI(fMRIs_path = 'fMRIs_shaefer1000_15364', subject=subject)[name] for subject in os.listdir('fMRIs_shaefer1000_15364') if subject[:3] == 'sub' and os.path.exists(f'fMRIs_shaefer1000_15364/{subject}/{movie}')]

    snr_values = calculate_snr(movies)
    mean_snr = aggregate_snr(snr_values)


    brain_path = '/media/miplab-nas2/Data2/Movies_Emo/Preprocessed_data/sub-S01/ses-1/pp_sub-S01_ses-1_YouAgain.feat/filtered_func_data_res_MNI.nii'
    mask3d_path = 'mask_schaefer1000_15364.npy'
    
    threshold = np.percentile(mean_snr, percentile)  # For percentile = 70, find the 70th percentile value
    relevant_voxels = np.where(mean_snr > threshold, 1, 0)

    mask_3d = np.load(mask3d_path, mmap_mode = 'r')
    mask_2d = mask_3d.reshape(-1,) 
    indices = np.where(mask_2d == 1)[0] # 15364 voxels

    brain_img = nib.load(brain_path)
    brain_data = np.asanyarray(brain_img.dataobj)[..., 50]

    # Step 1: Create a zero-filled array with the same shape as the original 3D data
    reconstructed_3d = np.zeros(mask_3d.shape, dtype=np.float32)
    
    # Step 2: Flatten the zero-filled array (this step is optional and for clarity)
    reconstructed_flat = reconstructed_3d.flatten()
    
    # Step 3: Use the stored indices to place the masked_fMRI values into the flattened array
    reconstructed_flat[indices] = relevant_voxels
    
    # Step 4: Reshape back to the original 3D shape
    reconstructed_3d = reconstructed_flat.reshape(mask_3d.shape)
    
    n_voxels = np.count_nonzero(reconstructed_3d)
    np.save(f'mask_schaefer1000_{n_voxels}', reconstructed_3d)

def calculate_snr(movies):
    # movies: dictionary where keys are movie identifiers and values are lists of arrays
    # each array can have a shape (time_points, voxels) corresponding to a single subject's data for that movie
    
    snr_movies = {}

    for movie_id, subject_data in movies.items():
        if not subject_data:
            continue  # skip if no data available for the movie

        # Stack subject arrays along a new subject dimension
        # Note: Ensure all arrays have the same second dimension (voxels)
        stacked_data = np.stack(subject_data, axis=0)

        num_subjects = stacked_data.shape[0]
        num_time_points = stacked_data.shape[1]
        num_voxels = stacked_data.shape[2]

        # Compute mean across subjects for each time point and voxel
        mean_subjects = np.mean(stacked_data, axis=0)
        
        # Compute signal variance across time (variance of means)
        signal_variance = np.var(mean_subjects, axis=0)
        
        # Compute mean across time for each subject and voxel
        mean_time = np.mean(stacked_data, axis=1)
        
        # Compute noise variance across subjects (variance of means)
        noise_variance = np.var(mean_time, axis=0)
        
        # Avoid division by zero in case of zero noise variance
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = signal_variance / noise_variance
            snr[noise_variance == 0] = 0  # set SNR to 0 where noise variance is 0
        
        snr_movies[movie_id] = snr

    return snr_movies

def aggregate_snr(snr_movies):
    # snr_movies: dictionary with movie IDs as keys and arrays of SNR values as values
    all_snrs = np.stack(list(snr_movies.values()), axis=0)  # Stack all SNR arrays along a new axis
    mean_snr = np.mean(all_snrs, axis=0)  # Compute the mean SNR across movies for each voxel
    return mean_snr

### DATA STORAGE ###

# def store_data():
#     """
#     Top-level function to initiate the storing of preprocessed fMRI and video data.
#     """
#     print('Start preprocessing and storing data. This can take some time.')
#     store_masked_fMRI()
#     print('All preprocessed fMRIs have been stored.')
#     store_processed_videos()
#     print('All preprocessed videos have been stored.')

# def store_masked_fMRI():
#     """
#     Store masked fMRI data from specified paths after applying a binary mask.
#     """
#     tic = time.time()
#     print('Start storing fMRIs.')
#     # you can change the path of the mask or the path of the fMRI data
#     mask_path = '/home/chchan/Michael-Nas2/ml-students2023/resources/vis_mask.nii'

#     # extract the mask and flatten it
#     img_mask = nib.load(mask_path)
#     mask_3d = np.asanyarray(img_mask.dataobj)
#     mask_2d = mask_3d.reshape(-1,)
#     # identify the relevant voxels to keep 
#     indices = np.where(mask_2d == 1)[0]
#     print(f'Using a mask covering {indices.shape[0]} voxels.')

#     # create the folder to store the data if it doesn't exist
#     os.makedirs('fMRIs', exist_ok=True)

#     data_path = '/media/miplab-nas2/Data2/Movies_Emo/Preprocessed_data'
#     for subject in os.listdir(data_path):
#         if subject[:5] == 'sub-S' and len(subject) == 7:
#             print(f'Preprocessing subject {subject}.')
#             subject_path = os.path.join(data_path, subject)
#             folder = os.path.join('fMRIs', subject)
#             os.makedirs(folder, exist_ok=True)

#             # you can change the for loop(s) depending on the structure of your folder 'data_path'
#             ses = ['ses-1', 'ses-2', 'ses-3', 'ses-4']
#             for s in ses:   
#                 feats = os.listdir(os.path.join(subject_path, s))[1:]
#                 for feat in feats:
#                     # extract the fMRI data
#                     if str(feat[17:-5]) != 'Rest':
#                         MNI_path = os.path.join(subject_path, s, feat, 'filtered_func_data_res_MNI.nii')
#                         name = MNI_path[85:-36]     # movie name
#                         img = nib.load(MNI_path)
#                         data_4d = np.asanyarray(img.dataobj)
        
#                         time_points = data_4d.shape[3]
#                         data_2d = data_4d.reshape(-1, time_points)     # flatten the fMRI data
#                         masked_fMRI = data_2d[indices].astype(np.float32)     # extract the relevant voxels only
        
#                         # save the file
#                         file_path = os.path.join(folder, f'{name}.npy')
#                         np.save(file_path, masked_fMRI)
#                         del img, data_4d, data_2d, masked_fMRI

#     store_average_fMRIs()

#     print("Storing fMRIs completed. Total time: {:.2f} minutes.".format((time.time() - tic) / 60))
#     print('---')

def store_masked_fMRI(mask_path, main_folder):
    """
    Store masked fMRI data from specified paths after applying a binary mask.
    """
    tic = time.time()
    print('Start storing fMRIs.')
    
    mask_3d = np.load(mask_path, mmap_mode = 'r')
    mask_2d = mask_3d.reshape(-1,)
    # identify the relevant voxels to keep 
    indices = np.where(mask_2d == 1)[0]
    print(f'Using a mask covering {indices.shape[0]} voxels.')

    # create the folder to store the data if it doesn't exist
    os.makedirs(main_folder, exist_ok=True)

    data_path = '/media/miplab-nas-shadow/Data2/Movies_Emo/Preprocessed_data'
    done = []
    for subject in os.listdir(data_path):
        if subject[:5] == 'sub-S' and len(subject) == 7 and int(subject[5:7]) not in done:
            print(f'Preprocessing subject {subject}.')
            subject_path = os.path.join(data_path, subject)
            folder = os.path.join(main_folder, subject)
            os.makedirs(folder, exist_ok=True)

            MNI_paths = list(Path(subject_path).rglob('filtered_func_data_res_MNI.nii'))
            for MNI_path in MNI_paths:
                name = str(MNI_path)[91:-36]     # movie name
                img = nib.load(MNI_path)
                data_4d = np.asanyarray(img.dataobj)

                time_points = data_4d.shape[3]
                data_2d = data_4d.reshape(-1, time_points)     # flatten the fMRI data
                masked_fMRI = data_2d[indices].astype(np.float32)     # extract the relevant voxels only

                # save the file
                file_path = os.path.join(folder, f'{name}.npy')
                np.save(file_path, masked_fMRI)
                del img, data_4d, data_2d, masked_fMRI

    store_average_fMRIs_new(main_folder)

    print("Storing fMRIs completed. Total time: {:.2f} minutes.".format((time.time() - tic) / 60))
    print('---')

# def store_average_fMRIs():
#     """
#     Compute and store the average fMRI data across subjects for each movie.
#     """
#     os.makedirs('fMRIs/average', exist_ok=True)
#     subject_list = os.listdir('fMRIs')
#     subject_list.remove('average')
#     for title in load_masked_fMRI(subject = 'sub-S01').keys():
#         fMRI_list = []
#         for subject in subject_list:
#             try:
#                 fMRI = load_masked_fMRI(subject = subject)[title]
#                 fMRI_list.append(fMRI)
#                 del fMRI
#             except KeyError:
#                 print(f"File '{title}' not found in subject '{subject}'. Skipping.")
#                 continue
#         average = np.mean(fMRI_list, axis = 0)
#         file_path = os.path.join('fMRIs/average', f'{title}.npy')
#         np.save(file_path, np.mean(fMRI_list, axis=0))
#         del average

def store_average_fMRIs(main_folder):
    """
    Compute and store the average fMRI data across subjects for each movie.
    """
    os.makedirs(main_folder + '/average', exist_ok=True)
    subject_list = os.listdir(main_folder)
    subject_list.remove('average')
    for title in load_masked_fMRI(fMRIs_path = main_folder, subject = 'sub-S01').keys():
        fMRI_list = []
        for subject in subject_list:
            try:
                fMRI = load_masked_fMRI(fMRIs_path = main_folder, subject = subject)[title]
                fMRI_list.append(fMRI)
                del fMRI
            except KeyError:
                print(f"File '{title}' not found in subject '{subject}'. Skipping.")
                continue
        average = np.mean(fMRI_list, axis = 0)
        file_path = os.path.join(main_folder + '/average', f'{title}.npy')
        np.save(file_path, np.mean(fMRI_list, axis=0))
        del average

def store_processed_videos():
    """
    Preprocess and store video data from specified paths.
    """
    tic = time.time()
    print('Start storing videos.')
    
    # you can change the path of the videos
    video_path = '/media/miplab-nas2/Data2/Michael/ml-students2023/data/Movies_cut'
    videos = os.listdir(video_path)

    # create the folder to store the data if it doesn't exist
    folder = 'processed_videos'
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    
    # number of workers for parallel processing
    num_workers = 4
    with ThreadPoolExecutor(max_workers=num_workers) as executor:

        for video in videos:

            # standardization of the movie names
            video_el = video.split('_')     # remove the '_'
            key_name = ''
            for el in video_el[:-1]:   # -1 to remove the extension .mp4
                key_name = str(key_name) + str(el)
                
            #some video names are not identical to the corresponding fMRI names
            if key_name == 'TearsofSteel':
                key_name = 'TearsOfSteel'
            elif key_name == 'Thesecretnumber':
                key_name = 'TheSecretNumber'

            print(f'Preprocessing video {key_name}.')

            # get the raw video
            video_reader = imageio.get_reader(os.path.join(video_path, video))
            
            # use executor for parallel processing to preprocess the video frames (resize, normalize)
            video_processed = list(executor.map(process_frame, video_reader))
            del video_reader

            # store the video in a numpy array
            video_array = np.array(video_processed)
            del video_processed

            # resample the videos (either upsampling or downsampling)
            n_frames = video_array.shape[0]
            _, movie_duration = load_durations(video = key_name, subject = 'S01')
            new_n_frames = movie_duration * 32     # 32 is the aimed frame rate (32 frames per TR). It can be modified
            indices = np.linspace(0, n_frames - 1, new_n_frames).astype(int)     # remove or duplicate some frames uniformly
            video_resampled = video_array[indices]
            del video_array

            num_channels = 3     # RGB
            # divide the video into smaller groups of 32 frames each
            video_reshaped = video_resampled.reshape(movie_duration, 32, 112, 112, num_channels)
            del video_resampled

            # transpose from shape (movie_duration, 32, 112, 112, num_channels) to shape (movie duration, num_channels, 112, 112, 32)
            video_transposed = np.transpose(video_reshaped, (0, 4, 2, 3, 1))
            del video_reshaped

            # save the file
            file_path = os.path.join(folder, f'{key_name}.npy')
            np.save(file_path, video_transposed)
            del video_transposed
    print("Storing videos completed. Total time: {:.2f} minutes.".format((time.time() - tic) / 60))
    print('---')

def normalize(X):
    """
    Normalize the input array or tensor to the range [0, 1].
    
    Arguments:
        X (np.ndarray or torch.Tensor): The data to normalize.
    
    Returns:
        np.ndarray or torch.Tensor: Normalized data.
    """
    epsilon = 1e-12      # small value to avoid division by zero
    if isinstance(X, np.ndarray):
        min_val = np.min(X)
        max_val = np.max(X)
    else: 
        min_val = torch.min(X)
        max_val = torch.max(X)
    return (X - min_val) / (max_val - min_val + epsilon)

def process_frame(frame):
    """
    Process a single video frame by resizing and normalizing it.
    
    Arguments:
        frame (np.ndarray): The video frame to process.
    
    Returns:
        np.ndarray: The processed video frame.
    """
    frame = frame.astype(np.float32)
    target_shape = (112, 112)     # the target shape (frame_length, frame_width) can be modified.
    resized_frame = resize(frame, target_shape, anti_aliasing=True)
    return normalize(resized_frame)


### DATA SPLITTING ###


def split_for_encoder(dataset_ID, test_size = 0.2, fMRIs_path = 'fMRIs', full_movie_test = 'YouAgain'):
    """
    Split data into training, validation, and test sets for the encoder model, and save them.
    By default, the last TRs of each movie constitute the test set, and the remaining TRs are splitted into the train and validation set
    sur that 1 every 5 TR go into the validation set. Also, the entire movie titled 'YouAgain' is added to the test set.
    
    Arguments:
        dataset_ID (str): Identifier for the dataset being used.
        test_size (float): Proportion of the data to be used as the test set.
        mask_size (int): Size of the fMRI mask in use.
    
    Effects:
        Creates and stores split data into designated subdirectories under a dataset-specific directory.
    """
    tic = time.time()
    print('Start splitting.')
    
    os.makedirs(f'encoder_dataset_{dataset_ID}/trainset', exist_ok=True)
    os.makedirs(f'encoder_dataset_{dataset_ID}/valset/videos', exist_ok=True)
    os.makedirs(f'encoder_dataset_{dataset_ID}/valset/fMRIs', exist_ok=True)
    os.makedirs(f'encoder_dataset_{dataset_ID}/testset/videos', exist_ok=True)
    os.makedirs(f'encoder_dataset_{dataset_ID}/testset/fMRIs', exist_ok=True)
    
    fMRIs, videos = load_data(fMRIs_path)
    video_titles = list(fMRIs.keys())
    mask_size = fMRIs[video_titles[0]].shape[1]
    
    train_id, val_id = {}, {}
    for key in video_titles:
        if key[0] != '.':
            TR = videos[key].shape[0]
            test_sep = int(TR*(1-test_size))
    
            if key == full_movie_test: #full movie to keep in the test set
                np.save(f'encoder_dataset_{dataset_ID}/testset/videos/{key}.npy', videos[key])
                np.save(f'encoder_dataset_{dataset_ID}/testset/fMRIs/{key}.npy', fMRIs[key])
                print(f'{key}: {TR}TRs. Train: 0%. Validation: 0%. Test: 100%.')
            else:
                np.save(f'encoder_dataset_{dataset_ID}/testset/videos/{key}.npy', videos[key][test_sep:, ...])
                np.save(f'encoder_dataset_{dataset_ID}/testset/fMRIs/{key}.npy', fMRIs[key][test_sep:, ...]) # take the last TRs for the test set
    
                video = videos[key][:test_sep, ...] # remove the frames that are part of the test set so they don't appear in the train and validation sets
            
                train_id[key] = [i for i in range(video.shape[0]) if i%5 != 0]
                val_id[key] = [i for i in range(video.shape[0]) if i%5 == 0]
                print(f'{key}: {TR}TRs. Train: {100*len(train_id[key])/TR:.2f}%. Validation: {100*len(val_id[key])/TR:.2f}%. Test: {100*(1 - test_sep/TR):.2f}%.')
        
    train_size = sum([len(train_id[key]) for key in train_id.keys()]) 
    train_label = np.memmap(f'encoder_dataset_{dataset_ID}/trainset/videos.npy', dtype='float32', mode='w+', shape=(train_size, 3, 112, 112, 32))
    train_input = np.memmap(f'encoder_dataset_{dataset_ID}/trainset/fMRIs.npy', dtype='float32', mode='w+', shape=(train_size, mask_size))

    start = 0
    for key in train_id.keys():
        print(f'Splitting {key}.')
        end = start + len(train_id[key])
        video = videos[key]
        train_label[start:end] = video[train_id[key]]
        train_input[start:end] = fMRIs[key][train_id[key]]
        start = end
        
        train_label.flush()
        train_input.flush()

        np.save(f'encoder_dataset_{dataset_ID}/valset/videos/{key}.npy', video[val_id[key]])
        np.save(f'encoder_dataset_{dataset_ID}/valset/fMRIs/{key}.npy', fMRIs[key][val_id[key]])

        del video

    del train_input, train_label, videos, fMRIs
        
    print("Splitting completed. Total time: {:.2f} minutes.".format((time.time() - tic) / 60))
    print('---')

def get_dataset(dataset_ID, mask_size = 4330):
    """
    Load the entire dataset already splitted into training, validation, and test sets from memory-mapped files.
    
    Arguments:
        dataset_ID (str): Identifier for the dataset directory.
        mask_size (int): Size of the fMRI mask in use.

    Returns:
        tuple: A tuple containing dictionaries for training, validation, and test sets.
           Each dictionary has keys 'fMRIs' and 'videos' pointing to the respective data.
    """
    trainset = {}
    trainset['fMRIs'] = np.memmap(f'encoder_dataset_{dataset_ID}/trainset/fMRIs.npy', dtype='float32', mode='r').reshape(-1, mask_size)
    trainset['videos'] = np.memmap(f'encoder_dataset_{dataset_ID}/trainset/videos.npy', dtype='float32', mode='r').reshape(-1, 3, 112, 112, 32)

    valset = {}
    fMRIs, videos = {}, {}
    path = f'encoder_dataset_{dataset_ID}/valset'
    for n in os.listdir(os.path.join(path, 'videos')):
        name = n[:-4]     # -4 to remove the extension '.npy' 
        f_path = os.path.join(path, f'fMRIs/{name}.npy')
        if os.path.exists(f_path):
            fMRIs[name] = np.load(f_path, mmap_mode = 'r')
        v_path = os.path.join(path, f'videos/{name}.npy')
        if os.path.exists(v_path):
            videos[name] = np.load(v_path, mmap_mode = 'r')
    valset['fMRIs'], valset['videos'] = fMRIs, videos

    testset = {}
    fMRIs, videos = {}, {}
    path = f'encoder_dataset_{dataset_ID}/testset'
    for n in os.listdir(os.path.join(path, 'videos')):
        name = n[:-4]     # -4 to remove the extension '.npy' 
        f_path = os.path.join(path, f'fMRIs/{name}.npy')
        if os.path.exists(f_path):
            fMRIs[name] = np.load(f_path, mmap_mode = 'r')
        v_path = os.path.join(path, f'videos/{name}.npy')
        if os.path.exists(v_path):
            videos[name] = np.load(v_path, mmap_mode = 'r')
    testset['fMRIs'], testset['videos'] = fMRIs, videos

    return trainset, valset, testset