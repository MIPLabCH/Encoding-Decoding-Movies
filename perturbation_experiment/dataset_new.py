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
#    print("\n\n\nfound it\n\n\n")
    #durations_path = '/media/miplab-nas-shadow/Data2/Michael-GOTO-RCPNAS/ml-students2023/resources/run_onsets.pkl'
    
    durations_path = '/media/RCPNAS/MIP/Michael/ml-students2023/resources/run_onsets.pkl'
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
                #data_2d shape:    (902629, 300)
                #indices shape:    (15000,)
                #masked_fMRI shape: (15000, 300)

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


'''
def store_middle_frame_original_size(video_path, output_path):
    """
    Preprocess and store video data from specified paths.
    """
    tic = time.time()
    print('Start storing videos.')
    
    # you can change the path of the videos
    video_path = '/media/miplab-nas2/Data2/Michael/ml-students2023/data/Movies_cut'
    videos = os.listdir(video_path)

    # create the folder to store the data if it doesn't exist
    folder = 'middle_frames_original_size'
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
            video_processed = list(executor.map(normalize_frame, video_reader))
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

            print("video_resampled shape:", video_resampled.shape)
            time.sleep(10)

            num_channels = 3     # RGB
            # divide the video into smaller groups of 32 frames each
            video_reshaped = video_resampled.reshape(movie_duration, 32, , , num_channels)
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
    return
'''



def store_middle_frame_original_size(video_path, output_path):
    """
    Preprocess and store video data from specified paths.
    """
    tic = time.time()
    print('Start storing videos.')
    
    # you can change the path of the videos
    video_path = '/media/RCPNAS/MIP/Michael/ml-students2023/data/Movies_cut'
    videos = os.listdir(video_path)

    # create the folder to store the data if it doesn't exist
    folder = 'middle_frames_original_size'
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    
    # number of workers for parallel processing
    num_workers = 4
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        
        for video in videos:
            
            if video != 'Superhero_exp.mp4':
                # standardization of the movie names
                video_el = video.split('_')
                # remove the '_'
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
                print("Got raw video. Total time: {:.2f} minutes. Now going to normalize frames.".format((time.time() - tic) / 60))

                            
                # use executor for parallel processing to preprocess the video frames (resize, normalize)
                #                           uncomment next lines

                #first_frame = video_reader.get_data(0)
                #height, width = first_frame.shape[:2]  # (height, width, channels))
                #total_frames = video_reader.get_length()
                
                #total_frames = 0
                #for frame in video_reader:
                #    total_frames += 1
                #print("Frames shape:", first_frame.shape)
                #print("Number of frames:", total_frames)

                
                video_processed = list(executor.map(process_frame_custom, video_reader))
                print("Resized and normalized frames. Total time: {:.2f} minutes. Now going to change video to numpy array.".format((time.time() - tic) / 60))
                del video_reader
                
                # store the video in a numpy array
                #video_array1 = np.array(video_reader[:video_reader.shape[0] // 2], :, :, :)  # process only first half to not kill the kernel
                #                           uncomment next line
                #for i in range(2):
                #    if i == 0:
                #        half1 = video_processed[:video_processed.shape[0] // 2, :, :, :]  # process only first half to not kill the kernel
                #        half2 = video_processed[video_processed.shape[0] // 2:, :, :, :]  # process only second half to not kill the kernel
                #        video_array = np.array(half1)
                #        print("Changed first half of video to numpy array. Total time: {:.2f} minutes. Now going to change frame rate.".format((time.time() - tic) / 60))
                #        del half1
                #    elif i == 1:
                #        video_array = np.array(half2)
                #        print("Changed second half of video to numpy array. Total time: {:.2f} minutes. Now going to change frame rate.".format((time.time() - tic) / 60))
                #        del half2
                    
                video_array = np.array(video_processed)
                print("Changed video to numpy array. Total time: {:.2f} minutes. Now going to change frame rate.".format((time.time() - tic) / 60))
                del video_processed
                
                # resample the videos (either upsampling or downsampling)
                n_frames = video_array.shape[0]
                _, movie_duration = load_durations(video = key_name, subject = 'S01')
                #new_n_frames = (movie_duration // 2) * 32  # this might be wrong, porbably the frames that are left out 
                new_n_frames = movie_duration * 32     # 32 is the aimed frame rate (32 frames per TR). It can be modified
                indices = np.linspace(0, n_frames - 1, new_n_frames).astype(int)     # remove or duplicate some frames uniformly
                video_resampled = video_array[indices]
                del video_array
                print("Resampled video. Total time: {:.2f} minutes. Now going to reshape video to sets of 32 frames.".format((time.time() - tic) / 60))
                
                num_channels = 3     # RGB

                print("video_resampled shape:", video_resampled.shape)
                time.sleep(10)
                
                # Get the original dimensions from the resampled video
                original_height, original_width = video_resampled.shape[1], video_resampled.shape[2]
                
                # divide the video into smaller groups of 32 frames each, keeping original resolution
                video_reshaped = video_resampled.reshape(movie_duration, 32, original_height, original_width, num_channels)
                del video_resampled
                
                # Extract just the middle frame (index 15, which is the 16th frame out of 32) from each group
                middle_frame_index = 15  # Middle frame from each set of 32 frames
                video_middle_frames = video_reshaped[:, middle_frame_index, :, :, :]  # shape: (movie_duration, height, width, num_channels)
                del video_reshaped
                
                # transpose from shape (movie_duration, height, width, num_channels) to shape (movie_duration, num_channels, height, width)
                video_transposed = np.transpose(video_middle_frames, (0, 3, 1, 2))
                del video_middle_frames
                
                # save the file
                file_path = os.path.join(folder, f'{key_name}.npy')
                np.save(file_path, video_transposed)
                del video_transposed
            
            else:
                print("skipped Superhero") #it's already there

    print("Storing videos completed. Total time: {:.2f} minutes.".format((time.time() - tic) / 60))
    print('---')
    return


'''
def store_middle_frame_original_size(video_path, output_path):
    """
    Preprocess and store video data from specified paths.
    """
    tic = time.time()
    print('Start storing videos.')
    
    # you can change the path of the videos
    video_path = '/media/RCPNAS/MIP/Michael/ml-students2023/data/Movies_cut'
    videos = os.listdir(video_path)

    # create the folder to store the data if it doesn't exist
    folder = 'middle_frames_original_size'
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    
    # number of workers for parallel processing
    num_workers = 4
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        
        for video in videos:
            
            if video != 'Superhero_exp.mp4':
                # standardization of the movie names
                video_el = video.split('_')
                # remove the '_'
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
                print("Got raw video. Total time: {:.2f} minutes. Now going to normalize frames.".format((time.time() - tic) / 60))
                           
                # use executor for parallel processing to preprocess the video frames (resize, normalize)
                video_processed = video_reader
#                video_processed = list(executor.map(normalize_frame, video_reader))
#                print("Normalized frames. Total time: {:.2f} minutes. Now going to change video to numpy array.".format((time.time() - tic) / 60))
#                del video_reader
                
                # CALCULATE RESAMPLING INDICES FOR THE FULL VIDEO FIRST
                total_frames = len(video_processed)
                _, movie_duration = load_durations(video=key_name, subject='S01')
                new_n_frames = movie_duration * 32  # Total frames needed for the desired duration
                
                # Calculate the resampling indices for the FULL video
                full_indices = np.linspace(0, total_frames - 1, new_n_frames).astype(int)
                
                print(f"Total frames: {total_frames}, Movie duration: {movie_duration} TRs")
                print(f"Will resample to {new_n_frames} frames using indices from {full_indices[0]} to {full_indices[-1]}")
                
                # Calculate split point
                half_point = len(video_processed) // 2
                
                # Determine which indices belong to each half
                indices_half1 = full_indices[full_indices < half_point]
                indices_half2 = full_indices[full_indices >= half_point] - half_point  # Adjust for the offset
                
                print(f"Half 1: {half_point} frames, using {len(indices_half1)} resampled frames")
                print(f"Half 2: {total_frames - half_point} frames, using {len(indices_half2)} resampled frames")
                
                # Process each half separately to save memory
                for i in range(2):
                    if i == 0:
                        print("Processing first half...")
                        half_data = video_processed[:half_point]
                        half_indices = indices_half1
                        half_name = "first"
                        half_suffix = "_half1"
                    elif i == 1:
                        print("Processing second half...")
                        half_data = video_processed[half_point:]
                        half_indices = indices_half2
                        half_name = "second"
                        half_suffix = "_half2"
                    
                    # Delete video_processed after extracting the half we need
                    if i == 0:
                        del video_processed  # Free memory after first extraction
                    
                    # Skip if no frames needed from this half
                    if len(half_indices) == 0:
                        print(f"No frames needed from {half_name} half, skipping...")
                        del half_data
                        continue
                    
                    # Convert to numpy array
                    video_array = np.array(half_data)
                    print("Changed {half_name} half to numpy array. Total time: {:.2f} minutes.".format((time.time() - tic) / 60))
                    del half_data  # Free memory immediately
                    
                    # Resample this half using the appropriate indices
                    video_resampled = video_array[half_indices]
                    del video_array
                    print("Resampled {half_name} half. Shape: {video_resampled.shape}. Total time: {:.2f} minutes.".format((time.time() - tic) / 60))
                    
                    num_channels = 3  # RGB
                    
                    print("video_resampled shape:", video_resampled.shape)
                    time.sleep(10)
                    
                    # Get the original dimensions from the resampled video
                    original_height, original_width = video_resampled.shape[1], video_resampled.shape[2]
                    
                    # Calculate how many TRs this half contributes
                    frames_in_half = video_resampled.shape[0]
                    trs_in_half = frames_in_half // 32  # Number of complete TRs
                    
                    # divide the video into smaller groups of 32 frames each, keeping original resolution
                    video_reshaped = video_resampled.reshape(trs_in_half, 32, original_height, original_width, num_channels)
                    del video_resampled
                    
                    # Extract just the middle frame (index 15, which is the 16th frame out of 32) from each group
                    middle_frame_index = 15  # Middle frame from each set of 32 frames
                    video_middle_frames = video_reshaped[:, middle_frame_index, :, :, :]  # shape: (trs_in_half, height, width, num_channels)
                    del video_reshaped
                    
                    # transpose from shape (trs_in_half, height, width, num_channels) to shape (trs_in_half, num_channels, height, width)
                    video_transposed = np.transpose(video_middle_frames, (0, 3, 1, 2))
                    del video_middle_frames
                    
                    # save the file for this half
                    file_path = os.path.join(folder, f'{key_name}{half_suffix}.npy')
                    np.save(file_path, video_transposed)
                    print(f"Saved {key_name}{half_suffix}.npy with shape {video_transposed.shape}")
                    del video_transposed
                    
            else:
                print("skipped Superhero")  # it's already there

    print("Storing videos completed. Total time: {:.2f} minutes.".format((time.time() - tic) / 60))
    print('---')
    return
'''





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
            _, movie_duration = load_durations(video = key_name, subject = 'S01')   #would anything change if we change the subject?
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
    #Processing frame with shape: {frame.shape}')
    target_shape = (112, 112)     # the target shape (frame_length, frame_width) can be modified.
    resized_frame = resize(frame, target_shape, anti_aliasing=True)
    return normalize(resized_frame)

def process_frame_custom(frame, width=None, height=None):
    """
    Process a single video frame by resizing and normalizing it.
    
    Arguments:
        frame (np.ndarray): The video frame to process.
    
    Returns:
        np.ndarray: The processed video frame.
    """
    frame = frame.astype(np.float32)
    #Processing frame with shape: {frame.shape}')
    if frame.shape[0] > 360 or frame.shape[1] > 640:
        target_shape = (360, 640)     # the target shape (frame_length, frame_width) can be modified.    
        resized_frame = resize(frame, target_shape, anti_aliasing=True)
    else:
        resized_frame = frame
    return normalize(resized_frame)


def normalize_frame(frame):
    """
    Process a single video frame by normalizing it.
    
    Arguments:
        frame (np.ndarray): The video frame to process.
    
    Returns:
        np.ndarray: The processed video frame.
    """
    frame = frame.astype(np.float32)
    return normalize(frame)


### DATA SPLITTING ###


def split_for_encoder(dataset_ID, test_size = 0.2, fMRIs_path = 'fMRIs', full_movie_test = 'YouAgain', subject='sub-S01'):
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
    
    fMRIs, videos = load_data(fMRIs_path, subject)
    video_titles = list(fMRIs.keys())
    mask_size = fMRIs[video_titles[0]].shape[1]
    
    train_id, val_id = {}, {}
    for key in video_titles:
        if key[0] != '.':
            TR = videos[key].shape[0]
            TR2 = fMRIs[key].shape[0]
            print("TR=", TR, "TR2=", TR2)
            if TR != TR2:
                print("\n\n\n\nyep big mistake, fMRI TRs =", str(TR2) + ", video TRs =", str(TR), "\n\n\n\n")
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



def split_for_decoder(dataset_ID, test_size=0.2, fMRIs_path='fMRIs', full_movie_test='YouAgain', subject='average'):
    """
    Split data into training, validation, and test sets for the decoder model, and save them.
    
    Arguments:
        dataset_ID (str): Identifier for the dataset being used.
        test_size (float): Proportion of the data to be used as the test set.
        fMRIs_path (str): Path to the directory containing preprocessed fMRI data.
        full_movie_test (str): Name of the movie to be fully included in the test set.
        subject (str): Specifies the subject to load data for. Default is 'average'.
    """
    tic = time.time()
    print('Start splitting.')
    
    # Create decoder dataset folders instead of encoder
    os.makedirs(f'decoder_dataset_{dataset_ID}/trainset', exist_ok=True)
    os.makedirs(f'decoder_dataset_{dataset_ID}/valset/videos', exist_ok=True)
    os.makedirs(f'decoder_dataset_{dataset_ID}/valset/fMRIs', exist_ok=True)
    os.makedirs(f'decoder_dataset_{dataset_ID}/testset/videos', exist_ok=True)
    os.makedirs(f'decoder_dataset_{dataset_ID}/testset/fMRIs', exist_ok=True)
    
    # Load data with specified subject
    fMRIs, videos = load_data(fMRIs_path, subject=subject)
    video_titles = list(fMRIs.keys())
    mask_size = fMRIs[video_titles[0]].shape[1]
    
    train_id, val_id = {}, {}
    for key in video_titles:
        if key[0] != '.':
            TR = videos[key].shape[0]
            test_sep = int(TR*(1-test_size))
    
            if key == full_movie_test: #full movie to keep in the test set
                np.save(f'decoder_dataset_{dataset_ID}/testset/videos/{key}.npy', videos[key])
                np.save(f'decoder_dataset_{dataset_ID}/testset/fMRIs/{key}.npy', fMRIs[key])
                print(f'{key}: {TR}TRs. Train: 0%. Validation: 0%. Test: 100%.')
            else:
                np.save(f'decoder_dataset_{dataset_ID}/testset/videos/{key}.npy', videos[key][test_sep:, ...])
                np.save(f'decoder_dataset_{dataset_ID}/testset/fMRIs/{key}.npy', fMRIs[key][test_sep:, ...]) # take the last TRs for the test set
    
                video = videos[key][:test_sep, ...] # remove the frames that are part of the test set
            
                train_id[key] = [i for i in range(video.shape[0]) if i%5 != 0]
                val_id[key] = [i for i in range(video.shape[0]) if i%5 == 0]
                print(f'{key}: {TR}TRs. Train: {100*len(train_id[key])/TR:.2f}%. Validation: {100*len(val_id[key])/TR:.2f}%. Test: {100*(1 - test_sep/TR):.2f}%.')
        
    train_size = sum([len(train_id[key]) for key in train_id.keys()]) 
    train_label = np.memmap(f'decoder_dataset_{dataset_ID}/trainset/videos.npy', dtype='float32', mode='w+', shape=(train_size, 3, 112, 112, 32))
    train_input = np.memmap(f'decoder_dataset_{dataset_ID}/trainset/fMRIs.npy', dtype='float32', mode='w+', shape=(train_size, mask_size))

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

        np.save(f'decoder_dataset_{dataset_ID}/valset/videos/{key}.npy', video[val_id[key]])
        np.save(f'decoder_dataset_{dataset_ID}/valset/fMRIs/{key}.npy', fMRIs[key][val_id[key]])

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



def get_dataset2(dataset_ID):
    trainset = {}
    trainset['fMRIs'] = np.load('processed_data_new/sub-S01/train.npy')
    #trainset['fMRIs'] = np.memmap(f'encoder_dataset_{dataset_ID}/trainset/fMRIs.npy', dtype='float32', mode='r').reshape(-1, mask_size)
    trainset['videos'] = np.memmap(f'encoder_dataset_{dataset_ID}/trainset/videos.npy', dtype='float32', mode='r').reshape(-1, 3, 112, 112, 32)


    valset = {}
    fMRIs, videos = {}, {}
    path = 'processed_data_new/sub-S01/val'
    for n in os.listdir(path):
        #print(os.listdir(path))
        name = n[:-4]     # -4 to remove the extension '.npy' 
        f_path = os.path.join(path, f'{name}.npy')
        if os.path.exists(f_path):
            fMRIs[name] = np.load(f_path)
        
        path = f'encoder_dataset_{dataset_ID}/valset'
        v_path = os.path.join(path, f'videos/{name}.npy')
        if os.path.exists(v_path):
            videos[name] = np.load(v_path, mmap_mode = 'r')
        path = 'processed_data_new/sub-S01/val'

    valset['fMRIs'], valset['videos'] = fMRIs, videos


    testset = {}
    fMRIs, videos = {}, {}
    path = 'processed_data_new/sub-S01/test'
    for n in os.listdir(path):
        name = n[:-4]     # -4 to remove the extension '.npy' 
        f_path = os.path.join(path, f'{name}.npy')
        if os.path.exists(f_path):
            fMRIs[name] = np.load(f_path)
        
        path = f'encoder_dataset_{dataset_ID}/testset'
        v_path = os.path.join(path, f'videos/{name}.npy')
        if os.path.exists(v_path):
            videos[name] = np.load(v_path, mmap_mode = 'r')
        path = 'processed_data_new/sub-S01/test'
    testset['fMRIs'], testset['videos'] = fMRIs, videos

    return trainset, valset, testset






def get_dataset3(dataset_ID):
    trainset = {}
    trainset['fMRIs'] = np.load('processed_data/sub-S01/train.npy')
    #trainset['fMRIs'] = np.memmap(f'encoder_dataset_{dataset_ID}/trainset/fMRIs.npy', dtype='float32', mode='r').reshape(-1, mask_size)
    trainset['videos'] = np.memmap(f'encoder_dataset_{dataset_ID}/trainset/videos.npy', dtype='float32', mode='r').reshape(-1, 3, 112, 112, 32)


    valset = {}
    '''
    fMRIs, videos = {}, {}
    path = 'processed_data_new/sub-S01/val'
    for n in os.listdir(path):
        #print(os.listdir(path))
        name = n[:-4]     # -4 to remove the extension '.npy' 
        f_path = os.path.join(path, f'{name}.npy')
        if os.path.exists(f_path):
            fMRIs[name] = np.load(f_path)
        
        path = f'encoder_dataset_{dataset_ID}/valset'
        v_path = os.path.join(path, f'videos/{name}.npy')
        if os.path.exists(v_path):
            videos[name] = np.load(v_path, mmap_mode = 'r')
        path = 'processed_data_new/sub-S01/val'

    valset['fMRIs'], valset['videos'] = fMRIs, videos
    '''


    testset = {}
    fMRIs, videos = {}, {}
    path = 'processed_data/sub-S01/test'
    for n in os.listdir(path):
        name = n[:-4]     # -4 to remove the extension '.npy' 
        f_path = os.path.join(path, f'{name}.npy')
        if os.path.exists(f_path):
            fMRIs[name] = np.load(f_path)
        
        path = f'encoder_dataset_{dataset_ID}/testset'
        v_path = os.path.join(path, f'videos/{name}.npy')
        if os.path.exists(v_path):
            videos[name] = np.load(v_path, mmap_mode = 'r')
        path = 'processed_data/sub-S01/test'
    testset['fMRIs'], testset['videos'] = fMRIs, videos

    return trainset, valset, testset



def get_dataset4():
    trainset = {}
    trainset['fMRIs'] = np.load('processed_data/sub-S01/train.npy')
    #trainset['fMRIs'] = np.memmap(f'encoder_dataset_{dataset_ID}/trainset/fMRIs.npy', dtype='float32', mode='r').reshape(-1, mask_size)
    trainset['videos'] = np.load('processed_data/videos/videos.npy')


    valset = {}
    '''
    fMRIs, videos = {}, {}
    path = 'processed_data_new/sub-S01/val'
    for n in os.listdir(path):
        #print(os.listdir(path))
        name = n[:-4]     # -4 to remove the extension '.npy' 
        f_path = os.path.join(path, f'{name}.npy')
        if os.path.exists(f_path):
            fMRIs[name] = np.load(f_path)
        
        path = f'encoder_dataset_{dataset_ID}/valset'
        v_path = os.path.join(path, f'videos/{name}.npy')
        if os.path.exists(v_path):
            videos[name] = np.load(v_path, mmap_mode = 'r')
        path = 'processed_data_new/sub-S01/val'

    valset['fMRIs'], valset['videos'] = fMRIs, videos
    '''


    testset = {}
    fMRIs, videos = {}, {}
    path = 'processed_data/sub-S01/test'
    for n in os.listdir(path):
        name = n[:-4]     # -4 to remove the extension '.npy' 
        f_path = os.path.join(path, f'{name}.npy')
        if os.path.exists(f_path):
            fMRIs[name] = np.load(f_path)
        
        path = 'processed_data/videos/test'
        v_path = os.path.join(path, f'{name}.npy')
        if os.path.exists(v_path):
            videos[name] = np.load(v_path)
        path = 'processed_data/sub-S01/test'
    testset['fMRIs'], testset['videos'] = fMRIs, videos

    return trainset, valset, testset









#functions for data analysis

#import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
#import os


def load_stats_data(pickle_path):
    """
    Load fMRI statistics from a pickle file and convert to a pandas DataFrame.
    
    Args:
        pickle_path (str): Path to the pickle file containing subject video statistics
        
    Returns:
        pandas.DataFrame: Processed DataFrame with subject, video, and statistics
    """
    # Load the statistics pickle file
    with open(pickle_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Convert the nested dictionary into a pandas DataFrame
    data = []
    for subject in stats:
        for video in stats[subject]:
            data.append({
                'subject': subject,
                'video': video,
                'fmri_mean': stats[subject][video]['fmri']['mean'],
                'fmri_var': stats[subject][video]['fmri']['var'],
                'fmri_min': stats[subject][video]['fmri']['min'],
                'fmri_max': stats[subject][video]['fmri']['max'],
                'video_mean': stats[subject][video]['video']['mean'],
                'video_var': stats[subject][video]['video']['var']
            })
    
    return pd.DataFrame(data)


def identify_outliers(df, group_col, metric_col):
    """
    Identify outliers in a DataFrame based on the 1.5 * IQR rule.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        group_col (str): Column name for grouping (e.g., 'subject' or 'video')
        metric_col (str): Column name for the metric to analyze (e.g., 'fmri_mean')
        
    Returns:
        dict: Dictionary with outliers for each group
    """
    outliers = {}
    for group in df[group_col].unique():
        subset = df[df[group_col] == group]
        q1 = subset[metric_col].quantile(0.25)
        q3 = subset[metric_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers for this group
        group_outliers = subset[(subset[metric_col] < lower_bound) | (subset[metric_col] > upper_bound)]
        if not group_outliers.empty:
            # If grouped by subject, return video and value; if grouped by video, return subject and value
            value_col = 'video' if group_col == 'subject' else 'subject'
            outliers[group] = group_outliers[[value_col, metric_col]].values.tolist()
    
    return outliers


def plot_boxplot_with_outliers(ax, x, y, data, title, ylabel):
    """
    Create a boxplot and identify outliers.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on
        x (str): Column name for x-axis (grouping variable)
        y (str): Column name for y-axis (metric to analyze)
        data (pandas.DataFrame): DataFrame containing the data
        title (str): Plot title
        ylabel (str): Y-axis label
        
    Returns:
        dict: Dictionary with outliers for each group
    """
    # Create boxplot
    sns.boxplot(x=x, y=y, data=data, ax=ax)
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x.capitalize(), fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Identify outliers
    return identify_outliers(data, x, y)


def create_boxplots_by_group(df, group_by='subject', save_path=None, figsize=(16, 20)):
    """
    Create boxplots for fMRI statistics grouped by subject or video.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        group_by (str): Column to group by, either 'subject' or 'video'
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
        figsize (tuple, optional): Figure size as (width, height)
        
    Returns:
        tuple: (Figure, Axes, Dictionary of outliers)
    """
    # Validate group_by parameter
    if group_by not in ['subject', 'video']:
        raise ValueError("group_by must be either 'subject' or 'video'")
    
    # Create figure with subplots for the four metrics
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # Prepare titles based on grouping
    group_name = group_by.capitalize()
    other_name = 'Video' if group_by == 'subject' else 'Subject'
    
    # Dictionary to store outliers
    all_outliers = {}
    
    # 1. Box plot for fMRI mean values
    all_outliers['mean'] = plot_boxplot_with_outliers(
        axes[0], group_by, 'fmri_mean', df, 
        f'Mean fMRI Values by {group_name} (Across All {other_name}s)', 
        'Mean'
    )
    
    # 2. Box plot for fMRI variance
    all_outliers['var'] = plot_boxplot_with_outliers(
        axes[1], group_by, 'fmri_var', df, 
        f'Variance of fMRI Values by {group_name} (Across All {other_name}s)', 
        'Variance'
    )
    
    # 3. Box plot for fMRI minimum values
    all_outliers['min'] = plot_boxplot_with_outliers(
        axes[2], group_by, 'fmri_min', df, 
        f'Minimum fMRI Values by {group_name} (Across All {other_name}s)', 
        'Minimum'
    )
    
    # 4. Box plot for fMRI maximum values
    all_outliers['max'] = plot_boxplot_with_outliers(
        axes[3], group_by, 'fmri_max', df, 
        f'Maximum fMRI Values by {group_name} (Across All {other_name}s)', 
        'Maximum'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes, all_outliers


def generate_summary_statistics(df, group_by='subject', save_path=None):
    """
    Generate summary statistics by subject or video and optionally save to CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        group_by (str): Column to group by, either 'subject' or 'video'
        save_path (str, optional): Path to save the summary CSV. If None, summary is not saved.
        
    Returns:
        pandas.DataFrame: Summary statistics
    """
    # Validate group_by parameter
    if group_by not in ['subject', 'video']:
        raise ValueError("group_by must be either 'subject' or 'video'")
    
    # Create summary
    summary = df.groupby(group_by).agg({
        'fmri_mean': ['mean', 'std', 'min', 'max'],
        'fmri_var': ['mean', 'std', 'min', 'max'],
        'fmri_min': ['mean', 'std', 'min', 'max'],
        'fmri_max': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    if save_path:
        summary.to_csv(save_path)
        print(f"Summary statistics saved to '{save_path}'")
    
    return summary


def print_outliers(outliers_dict, metric_name):
    """
    Print outliers in a readable format.
    
    Args:
        outliers_dict (dict): Dictionary with outliers
        metric_name (str): Name of the metric being analyzed
    """
    print(f"\n{metric_name} Outliers:")
    if not outliers_dict:
        print("No outliers found")
        return
    
    for group, outliers in outliers_dict.items():
        print(f"\n{group}:")
        for outlier in outliers:
            item, value = outlier
            print(f"  - {item}: {value:.6f}")


def analyze_fmri_statistics(pickle_path, group_by='subject', output_dir='outputs', save_plots=True, save_summary=True, display_plots=True):
    """
    Comprehensive analysis of fMRI statistics from a pickle file.
    
    Args:
        pickle_path (str): Path to the pickle file
        group_by (str): Column to group by, either 'subject' or 'video'
        output_dir (str): Directory to save outputs
        save_plots (bool): Whether to save the plots
        save_summary (bool): Whether to save the summary statistics
        display_plots (bool): Whether to display the plots
        
    Returns:
        tuple: (DataFrame, summary DataFrame, outliers dictionary)
    """
    # Validate group_by parameter
    if group_by not in ['subject', 'video']:
        raise ValueError("group_by must be either 'subject' or 'video'")
    
    # Create output directory if needed
    if (save_plots or save_summary) and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    df = load_stats_data(pickle_path)
    
    # Set paths for saving outputs
    if save_plots:
        plot_path = os.path.join(output_dir, f'fmri_statistics_by_{group_by}_boxplots.png')
    else:
        plot_path = None
    
    if save_summary:
        summary_path = os.path.join(output_dir, f'fmri_statistics_summary_by_{group_by}.csv')
    else:
        summary_path = None
    
    # Create visualization
    fig, axes, outliers = create_boxplots_by_group(df, group_by=group_by, save_path=plot_path)
    
    # Generate summary statistics
    summary = generate_summary_statistics(df, group_by=group_by, save_path=summary_path)
    
    # Print summary and outliers information
    print(f"Summary statistics by {group_by}:")
    print(summary)
    
    print("\n=== OUTLIERS INFORMATION ===")
    print_outliers(outliers['mean'], "Mean fMRI Values")
    print_outliers(outliers['var'], "Variance of fMRI Values")
    print_outliers(outliers['min'], "Minimum fMRI Values")
    print_outliers(outliers['max'], "Maximum fMRI Values")
    
    # Display plot if requested
    if display_plots:
        plt.show()
    else:
        plt.close(fig)
    
    return df, summary, outliers




'''
def create_dataset_all_subjects(subjects_folder):

    # Set the path to the main directory
    main_dir = subjects_folder

    # Find all subject directories (excluding specified subjects)
    excluded_subjects = ['sub-S02', 'sub-S21', 'sub-S26', 'sub-S30']
    subject_dirs = sorted([d for d in os.listdir(main_dir) 
                        if d.startswith('sub-') and d not in excluded_subjects])

    # Process each subject
    for subject in subject_dirs:
        subject_path = os.path.join(main_dir, subject)
        
        # Skip if not a directory
        if not os.path.isdir(subject_path):
            continue
        
        print(f"Processing {subject}...")
        
        # Get all .npy files for this subject
        npy_files = [f for f in os.listdir(subject_path) if f.endswith('.npy')]
        
        # Process each video file individually
        for npy_file in npy_files:
            file_path = os.path.join(subject_path, npy_file)
            video_name = os.path.splitext(npy_file)[0]  # Remove .npy extension to get video name
            
            try:
                # Load the data
                data = np.load(file_path)

                print("\n\nIt will load_durations with subject=" + str(subject), "\n\n")

                # taking care of onset of fmri to make sure shapes between fmri and video frames match
                onset, duration = load_durations(video=video_name, subject=subject)
                    
                # Apply necessary slicing and transposing
#                fmri_shape_before = fmri_data.shape
                data = data[:,onset:onset+duration]
                data = np.transpose(data,(1,0))
                
                # Z-score normalize the data
                # Using the formula: data = (data - np.mean(data)) / np.std(data)
                normalized_data = (data - np.mean(data)) / np.std(data)
                #normalized_data = data
                
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
'''



from time import sleep

'''def create_dataset_all_subjects(subjects_folder, output_dir="processed_data"):

    sleep(10)
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the path to the main directory
    main_dir = subjects_folder

    # List of subjects with known data issues (will be processed but marked)
    subjects_with_issues = ['sub-S02', 'sub-S21', 'sub-S26', 'sub-S30']
    
    # Find all subject directories
    #subject_dirs = sorted([d for d in os.listdir(main_dir) if (d.startswith('sub-') or d.startswith('average'))])
    subject_dirs = sorted([d for d in os.listdir(main_dir) if (d.startswith('average'))])

    # Dictionaries to track all processed data
    all_train_data = {}
    all_test_data = {}
    test_movie = "YouAgain.npy"  # Movie reserved entirely for test set
    
    # Process each subject
    for subject in subject_dirs:
        subject_path = os.path.join(main_dir, subject)
        
        # Skip if not a directory
        if not os.path.isdir(subject_path):
            continue
        
        if subject in subjects_with_issues:
            print(f"Processing {subject} (NOTE: This subject has known data issues)...")
        else:
            print(f"Processing {subject}...")
        
        # Create output directories for this subject
        subject_output_dir = os.path.join(output_dir, subject)
        subject_test_dir = os.path.join(subject_output_dir, "test")
        os.makedirs(subject_output_dir, exist_ok=True)
        os.makedirs(subject_test_dir, exist_ok=True)
        
        # Get all .npy files for this subject
        npy_files = [f for f in os.listdir(subject_path) if f.endswith('.npy')]
        
        # Initialize arrays to collect this subject's training data
        subject_train_data = []
        
        # Dictionary to track test data for this subject by movie
        subject_test_data = {}
        
        # Process each video file individually
        for npy_file in npy_files:
            file_path = os.path.join(subject_path, npy_file)
            video_name = os.path.splitext(npy_file)[0]  # Remove .npy extension to get video name
            
            try:
                # Load the data
                data = np.load(file_path)

                subject_id_short = subject[4:] if subject.startswith('sub-') else subject

                print(f"  Processing {video_name} for {subject_id_short}")

                print("\n\nIt will load_durations with subject=" + str(subject_id_short), "\n\n")

                if subject[:4] == 'sub-':
                    # Taking care of onset of fMRI to make sure shapes between fMRI and video frames match
                    onset, duration = load_durations(video=video_name, subject=subject_id_short)

                    print("if it prints this, the error was not in load_durations")
                    
                    # Apply necessary slicing and transposing
                    data = data[:, onset:onset+duration]
                    data = np.transpose(data, (1, 0))
                
                # Z-score normalize the data
                # normalized_data = (data - np.mean(data)) / np.std(data)
                normalized_data = data
                
                # Determine if this movie is for test set or needs to be split
                if npy_file == test_movie:
                    # Save entire movie to test data
                    test_output_path = os.path.join(subject_test_dir, f"{video_name}.npy")
                    np.save(test_output_path, normalized_data)
                    
                    subject_test_data[video_name] = normalized_data.shape
                    print(f"    Saved {video_name} to test set, shape: {normalized_data.shape}")
                else:
                    # Split into train (80%) and test (20%)
                    split_idx = int(normalized_data.shape[0] * 0.8)
                    train_part = normalized_data[:split_idx]
                    test_part = normalized_data[split_idx:]
                    
                    # Add training part to subject's training data
                    subject_train_data.append(train_part)
                    
                    # Save test part
                    test_output_path = os.path.join(subject_test_dir, f"{video_name}.npy")
                    np.save(test_output_path, test_part)
                    
                    subject_test_data[video_name] = test_part.shape
                    print(f"    Added {video_name} to train set, shape: {train_part.shape}")
                    print(f"    Saved {video_name} test portion to test set, shape: {test_part.shape}")
                
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        
        # Concatenate all training data for this subject
        if subject_train_data:
            subject_train_combined = np.concatenate(subject_train_data, axis=0)
            train_output_path = os.path.join(subject_output_dir, "train.npy")
            np.save(train_output_path, subject_train_combined)
            
            all_train_data[subject] = subject_train_combined.shape
            all_test_data[subject] = subject_test_data
            
            print(f"  Saved {subject} train data, shape: {subject_train_combined.shape}")
            
            # Create a note file for subjects with known issues
            if subject in subjects_with_issues:
                note_path = os.path.join(subject_output_dir, "NOTE_DATA_ISSUES.txt")
                with open(note_path, 'w') as f:
                    f.write(f"This subject ({subject}) has known data issues. Use with caution in your analysis.")
    
    # Print summary of all data shapes
    print("\nSummary of all train data shapes:")
    for subject, shape in all_train_data.items():
        if subject in subjects_with_issues:
            print(f"{subject} (NOTE: Has data issues): {shape}")
        else:
            print(f"{subject}: {shape}")
    
    print("\nSummary of all test data shapes by subject and movie:")
    for subject, movies in all_test_data.items():
        if subject in subjects_with_issues:
            print(f"\n{subject} (NOTE: Has data issues):")
        else:
            print(f"\n{subject}:")
        for movie, shape in movies.items():
            print(f"  {movie}: {shape}")
    
    return all_train_data, all_test_data'''



#import os
#import numpy as np
import json
from datetime import datetime
#from time import sleep

def create_dataset_all_subjects(subjects_folder, output_dir="processed_data"):
    sleep(10)
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the path to the main directory
    main_dir = subjects_folder

    # List of subjects with known data issues (will be processed but marked)
    subjects_with_issues = ['sub-S02', 'sub-S21', 'sub-S26', 'sub-S30']
    
    # Find all subject directories
    #subject_dirs = sorted([d for d in os.listdir(main_dir) if (d.startswith('sub-') or d.startswith('average'))])
    subject_dirs = sorted([d for d in os.listdir(main_dir) if (d.startswith('sub-S01') or d.startswith('average'))])
    #subject_dirs = sorted([d for d in os.listdir(main_dir) if (d.startswith('sub-S01'))])

    # Dictionaries to track all processed data
    all_train_data = {}
    all_test_data = {}
    test_movie = "YouAgain.npy"  # Movie reserved entirely for test set
    
    # Process each subject
    for subject in subject_dirs:
        subject_path = os.path.join(main_dir, subject)
        
        # Skip if not a directory
        if not os.path.isdir(subject_path):
            continue
        
        if subject in subjects_with_issues:
            print(f"Processing {subject} (NOTE: This subject has known data issues)...")
        else:
            print(f"Processing {subject}...")
        
        # Create output directories for this subject
        subject_output_dir = os.path.join(output_dir, subject)
        subject_test_dir = os.path.join(subject_output_dir, "test")
        os.makedirs(subject_output_dir, exist_ok=True)
        os.makedirs(subject_test_dir, exist_ok=True)
        
        # Get all .npy files for this subject
        npy_files = [f for f in os.listdir(subject_path) if f.endswith('.npy')]
        
        # Initialize arrays to collect this subject's training data
        subject_train_data = []
        
        # Dictionary to track test data for this subject by movie
        subject_test_data = {}
        
        # Create a dictionary to track processing order and details for this subject
        processing_log = {
            "subject": subject,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_order": [],
            "test_movies": {},
            "total_train_TRs": 0
        }
        
        # Track the cumulative index for data points in the concatenated training file
        cumulative_index = 0
        total_train_TRs = 0
        
        # Process each video file individually
        for npy_file in npy_files:
            file_path = os.path.join(subject_path, npy_file)
            video_name = os.path.splitext(npy_file)[0]  # Remove .npy extension to get video name
            
            try:
                # Load the data
                data = np.load(file_path)

                subject_id_short = subject[4:] if subject.startswith('sub-') else subject

                print(f"  Processing {video_name} for {subject_id_short}")

                print("\n\nIt will load_durations with subject=" + str(subject_id_short), "\n\n")

                if subject[:4] == 'sub-':
                    # Taking care of onset of fMRI to make sure shapes between fMRI and video frames match
                    onset, duration = load_durations(video=video_name, subject=subject_id_short)

                    print("if it prints this, the error was not in load_durations")
                    
                    # Apply necessary slicing and transposing
                    data = data[:, onset:onset+duration]
                    data = np.transpose(data, (1, 0))
                
                # Z-score normalize the data
                normalized_data = (data - np.mean(data)) / np.std(data)
                # normalized_data = data
                
                # Determine if this movie is for test set or needs to be split
                if npy_file == test_movie:
                    # Save entire movie to test data
                    test_output_path = os.path.join(subject_test_dir, f"{video_name}.npy")
                    np.save(test_output_path, normalized_data)
                    
                    test_TRs = normalized_data.shape[0]
                    subject_test_data[video_name] = normalized_data.shape
                    
                    # Log test movie details
                    processing_log["test_movies"][video_name] = {
                        "type": "full_movie",
                        "shape": normalized_data.shape,
                        "TRs": test_TRs
                    }
                    
                    print(f"    Saved {video_name} to test set, shape: {normalized_data.shape}, TRs: {test_TRs}")
                else:
                    # Split into train (80%) and test (20%)
                    split_idx = int(normalized_data.shape[0] * 0.8)
                    train_part = normalized_data[:split_idx]
                    test_part = normalized_data[split_idx:]
                    
                    # Calculate TRs
                    train_TRs = train_part.shape[0]
                    test_TRs = test_part.shape[0]
                    
                    # Update total train TRs
                    total_train_TRs += train_TRs
                    
                    # Add training part to subject's training data
                    subject_train_data.append(train_part)
                    
                    # Log the training portion details
                    processing_log["train_order"].append({
                        "movie": video_name,
                        "portion": "first_80_percent",
                        "train_shape": train_part.shape,
                        "TRs": train_TRs,
                        "start_index_in_combined": cumulative_index,
                        "end_index_in_combined": cumulative_index + train_part.shape[0] - 1
                    })
                    
                    # Update the cumulative index
                    cumulative_index += train_part.shape[0]
                    
                    # Save test part
                    test_output_path = os.path.join(subject_test_dir, f"{video_name}.npy")
                    np.save(test_output_path, test_part)
                    
                    subject_test_data[video_name] = test_part.shape
                    
                    # Log test portion details
                    processing_log["test_movies"][video_name] = {
                        "type": "last_20_percent",
                        "shape": test_part.shape,
                        "TRs": test_TRs
                    }
                    
                    print(f"    Added {video_name} to train set, shape: {train_part.shape}, TRs: {train_TRs}")
                    print(f"    Saved {video_name} test portion to test set, shape: {test_part.shape}, TRs: {test_TRs}")
                
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        
        # Concatenate all training data for this subject
        if subject_train_data:
            subject_train_combined = np.concatenate(subject_train_data, axis=0)
            train_output_path = os.path.join(subject_output_dir, "train.npy")
            np.save(train_output_path, subject_train_combined)
            
            all_train_data[subject] = subject_train_combined.shape
            all_test_data[subject] = subject_test_data
            
            # Update processing log with combined shape
            processing_log["combined_train_shape"] = subject_train_combined.shape
            processing_log["total_train_TRs"] = total_train_TRs
            
            print(f"  Saved {subject} train data, shape: {subject_train_combined.shape}, Total TRs: {total_train_TRs}")
            
            # Save the processing log to JSON file for this subject
            log_path = os.path.join(subject_output_dir, "processing_order.json")
            with open(log_path, 'w') as f:
                json.dump(processing_log, f, indent=2)
            
            # Also save a more readable text version
            log_txt_path = os.path.join(subject_output_dir, "processing_order.txt")
            with open(log_txt_path, 'w') as f:
                f.write(f"fMRI Processing Order Log for {subject} - {processing_log['timestamp']}\n")
                f.write(f"\nCombined Training Data Shape: {processing_log['combined_train_shape']}\n")
                f.write(f"Total Training TRs: {processing_log['total_train_TRs']}\n")
                f.write("\nOrder of Movies in Training Data:\n")
                f.write("==============================\n")
                
                for idx, movie_info in enumerate(processing_log["train_order"]):
                    f.write(f"{idx+1}. {movie_info['movie']}\n")
                    f.write(f"   Training portion shape: {movie_info['train_shape']}\n")
                    f.write(f"   TRs: {movie_info['TRs']}\n")
                    f.write(f"   Position in combined file: {movie_info['start_index_in_combined']} to {movie_info['end_index_in_combined']}\n\n")
                
                f.write("\nTest Movies:\n")
                f.write("===========\n")
                for movie, info in processing_log["test_movies"].items():
                    f.write(f"{movie} - {info['type']}, Shape: {info['shape']}, TRs: {info['TRs']}\n")
            
            # Create a note file for subjects with known issues
            if subject in subjects_with_issues:
                note_path = os.path.join(subject_output_dir, "NOTE_DATA_ISSUES.txt")
                with open(note_path, 'w') as f:
                    f.write(f"This subject ({subject}) has known data issues. Use with caution in your analysis.")
    
    # Print summary of all data shapes
    print("\nSummary of all train data shapes:")
    for subject, shape in all_train_data.items():
        if subject in subjects_with_issues:
            print(f"{subject} (NOTE: Has data issues): {shape}")
        else:
            print(f"{subject}: {shape}")
    
    print("\nSummary of all test data shapes by subject and movie:")
    for subject, movies in all_test_data.items():
        if subject in subjects_with_issues:
            print(f"\n{subject} (NOTE: Has data issues):")
        else:
            print(f"\n{subject}:")
        for movie, shape in movies.items():
            print(f"  {movie}: {shape}")
    
    return all_train_data, all_test_data




#import os
#import numpy as np
import json
from datetime import datetime

def create_video_dataset(videos_folder, output_dir="processed_data"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    video_train_dir = os.path.join(output_dir, "videos")
    video_test_dir = os.path.join(output_dir, "videos", "test")
    os.makedirs(video_train_dir, exist_ok=True)
    os.makedirs(video_test_dir, exist_ok=True)
    
    # Set the path to the videos directory
    videos_dir = videos_folder
    
    # Define the exact order of videos to process
    video_order = [
        'Sintel',
        'Payload',
        'TearsOfSteel',
        'Superhero',
        'BigBuckBunny',
        'FirstBite',
        'BetweenViewings',
        'AfterTheRain',
        'TheSecretNumber',
        'Chatter',
        'Spaceman',
        'LessonLearned',
        'YouAgain',
        'ToClaireFromSonny'
    ]
    
    # Test movie to be entirely in test set
    test_movie = "YouAgain"
    
    # Initialize lists to collect training video data
    train_videos_data = []
    
    # Dictionary to track all processed data shapes
    train_data_shape = None
    test_data_shapes = {}
    
    # Create a dictionary to track processing order and details
    processing_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_order": [],
        "test_movies": {},
        "total_train_TRs": 0
    }
    
    # Track the cumulative index for data points in the concatenated training file
    cumulative_index = 0
    total_train_TRs = 0
    
    # Process each video in the specified order
    for video_name in video_order:
        # Skip files starting with ForrestGump
        if video_name.startswith('ForrestGump'):
            continue
            
        # Construct file path with .npy extension
        file_path = os.path.join(videos_dir, f"{video_name}.npy")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue
        
        try:
            # Load the video data
            video_data = np.load(file_path)
            
            print(f"Processing {video_name}, shape: {video_data.shape}")
            
            # Determine if this is the test movie or needs to be split
            if video_name == test_movie:
                # Save entire movie to test data
                test_output_path = os.path.join(video_test_dir, f"{video_name}.npy")
                np.save(test_output_path, video_data)
                
                test_data_shapes[video_name] = video_data.shape
                test_TRs = video_data.shape[0]
                processing_log["test_movies"][video_name] = {
                    "type": "full_movie",
                    "shape": video_data.shape,
                    "frames": video_data.shape[0],
                    "TRs": test_TRs
                }
                
                print(f"  Saved {video_name} to test set, shape: {video_data.shape}, TRs: {test_TRs}")
            else:
                # Split into train (80%) and test (20%)
                split_idx = int(video_data.shape[0] * 0.8)
                train_part = video_data[:split_idx]
                test_part = video_data[split_idx:]
                
                # Calculate TRs
                train_TRs = train_part.shape[0]
                test_TRs = test_part.shape[0]
                
                # Update total train TRs
                total_train_TRs += train_TRs
                
                # Add training part to collection
                train_videos_data.append(train_part)
                
                # Log the training portion details
                processing_log["train_order"].append({
                    "movie": video_name,
                    "portion": "first_80_percent",
                    "original_shape": video_data.shape,
                    "train_shape": train_part.shape,
                    "frame_count": train_part.shape[0],
                    "TRs": train_TRs,
                    "start_index_in_combined": cumulative_index,
                    "end_index_in_combined": cumulative_index + train_part.shape[0] - 1
                })
                
                # Update the cumulative index
                cumulative_index += train_part.shape[0]
                
                # Save test part
                test_output_path = os.path.join(video_test_dir, f"{video_name}.npy")
                np.save(test_output_path, test_part)
                
                test_data_shapes[video_name] = test_part.shape
                processing_log["test_movies"][video_name] = {
                    "type": "last_20_percent",
                    "shape": test_part.shape,
                    "frames": test_part.shape[0],
                    "TRs": test_TRs
                }
                
                print(f"  Added {video_name} to train set, shape: {train_part.shape}, TRs: {train_TRs}")
                print(f"  Saved {video_name} test portion to test set, shape: {test_part.shape}, TRs: {test_TRs}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Concatenate all training videos
    if train_videos_data:
        train_videos_combined = np.concatenate(train_videos_data, axis=0)
        train_output_path = os.path.join(video_train_dir, "videos.npy")
        np.save(train_output_path, train_videos_combined)
        
        train_data_shape = train_videos_combined.shape
        processing_log["combined_train_shape"] = train_data_shape
        processing_log["total_train_TRs"] = total_train_TRs
        
        print(f"Saved combined training videos, shape: {train_data_shape}, Total TRs: {total_train_TRs}")
    
    # Save the processing log to a JSON file
    log_path = os.path.join(video_train_dir, "processing_order.json")
    with open(log_path, 'w') as f:
        json.dump(processing_log, f, indent=2)
    
    # Also save a more readable text version
    log_txt_path = os.path.join(video_train_dir, "processing_order.txt")
    with open(log_txt_path, 'w') as f:
        f.write(f"Video Processing Order Log - {processing_log['timestamp']}\n")
        f.write(f"\nCombined Training Data Shape: {processing_log['combined_train_shape']}\n")
        f.write(f"Total Training TRs: {processing_log['total_train_TRs']}\n")
        f.write("\nOrder of Movies in Training Data:\n")
        f.write("==============================\n")
        
        for idx, movie_info in enumerate(processing_log["train_order"]):
            f.write(f"{idx+1}. {movie_info['movie']}\n")
            f.write(f"   Original shape: {movie_info['original_shape']}\n")
            f.write(f"   Training portion shape: {movie_info['train_shape']}\n")
            f.write(f"   Frames: {movie_info['frame_count']}\n")
            f.write(f"   TRs: {movie_info['TRs']}\n")
            f.write(f"   Position in combined file: {movie_info['start_index_in_combined']} to {movie_info['end_index_in_combined']}\n\n")
        
        f.write("\nTest Movies:\n")
        f.write("===========\n")
        for movie, info in processing_log["test_movies"].items():
            f.write(f"{movie} - {info['type']}, Shape: {info['shape']}, Frames: {info['frames']}, TRs: {info['TRs']}\n")
    
    # Print summary of all data shapes
    print("\nSummary of data shapes:")
    print(f"Training videos: {train_data_shape}, Total TRs: {total_train_TRs}")
    
    print("\nTest videos by movie:")
    for movie, shape in test_data_shapes.items():
        print(f"  {movie}: {shape}, TRs: {processing_log['test_movies'][movie]['TRs']}")
    
    print(f"\nProcessing order has been saved to {log_path} and {log_txt_path}")
    
    return train_data_shape, test_data_shapes

# Example usage:
# train_shape, test_shapes = create_video_dataset("path/to/processed_videos", "path/to/processed_data")



'''
def create_video_dataset(videos_folder, output_dir="processed_data"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    video_train_dir = os.path.join(output_dir, "videos")
    video_test_dir = os.path.join(output_dir, "videos", "test")
    os.makedirs(video_train_dir, exist_ok=True)
    os.makedirs(video_test_dir, exist_ok=True)
    
    # Set the path to the videos directory
    videos_dir = videos_folder
    
    # Define the exact order of videos to process
    video_order = [
        'Sintel',
        'Payload',
        'TearsOfSteel',
        'Superhero',
        'BigBuckBunny',
        'FirstBite',
        'BetweenViewings',
        'AfterTheRain',
        'TheSecretNumber',
        'Chatter',
        'Spaceman',
        'LessonLearned',
        'YouAgain',
        'ToClaireFromSonny'
    ]
    
    # Test movie to be entirely in test set
    test_movie = "YouAgain"
    
    # Initialize lists to collect training video data
    train_videos_data = []
    
    # Dictionary to track all processed data shapes
    train_data_shape = None
    test_data_shapes = {}
    
    # Process each video in the specified order
    for video_name in video_order:
        # Skip files starting with ForrestGump
        if video_name.startswith('ForrestGump'):
            continue
            
        # Construct file path with .npy extension
        file_path = os.path.join(videos_dir, f"{video_name}.npy")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Skipping.")
            continue
        
        try:
            # Load the video data
            video_data = np.load(file_path)
            
            print(f"Processing {video_name}, shape: {video_data.shape}")
            
            # Determine if this is the test movie or needs to be split
            if video_name == test_movie:
                # Save entire movie to test data
                test_output_path = os.path.join(video_test_dir, f"{video_name}.npy")
                np.save(test_output_path, video_data)
                
                test_data_shapes[video_name] = video_data.shape
                print(f"  Saved {video_name} to test set, shape: {video_data.shape}")
            else:
                # Split into train (80%) and test (20%)
                split_idx = int(video_data.shape[0] * 0.8)
                train_part = video_data[:split_idx]
                test_part = video_data[split_idx:]
                
                # Add training part to collection
                train_videos_data.append(train_part)
                
                # Save test part
                test_output_path = os.path.join(video_test_dir, f"{video_name}.npy")
                np.save(test_output_path, test_part)
                
                test_data_shapes[video_name] = test_part.shape
                print(f"  Added {video_name} to train set, shape: {train_part.shape}")
                print(f"  Saved {video_name} test portion to test set, shape: {test_part.shape}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Concatenate all training videos
    if train_videos_data:
        train_videos_combined = np.concatenate(train_videos_data, axis=0)
        train_output_path = os.path.join(video_train_dir, "videos.npy")
        np.save(train_output_path, train_videos_combined)
        
        train_data_shape = train_videos_combined.shape
        print(f"Saved combined training videos, shape: {train_data_shape}")
    
    # Print summary of all data shapes
    print("\nSummary of data shapes:")
    print(f"Training videos: {train_data_shape}")
    
    print("\nTest videos by movie:")
    for movie, shape in test_data_shapes.items():
        print(f"  {movie}: {shape}")
    
    return train_data_shape, test_data_shapes
'''
    
# Example usage:
# train_shape, test_shapes = create_video_dataset("path/to/processed_videos", "path/to/processed_data")

# Example usage:
# train_shape, test_shapes = create_video_dataset("path/to/processed_videos", "path/to/processed_data")









'''
#functions for data analysis

import matplotlib.pyplot as plt
import pickle
import seaborn as sns

def load_and_process_data(pickle_path):
    """
    Load the statistics pickle file and convert it to a pandas DataFrame.
    
    Args:
        pickle_path (str): Path to the pickle file containing subject video statistics
        
    Returns:
        pandas.DataFrame: Processed DataFrame with subject, video, and statistics
    """
    # Load the statistics pickle file
    with open(pickle_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Convert the nested dictionary into a pandas DataFrame
    data = []
    for subject in stats:
        for video in stats[subject]:
            data.append({
                'subject': subject,
                'video': video,
                'fmri_mean': stats[subject][video]['fmri']['mean'],
                'fmri_var': stats[subject][video]['fmri']['var'],
                'fmri_min': stats[subject][video]['fmri']['min'],
                'fmri_max': stats[subject][video]['fmri']['max'],
                'video_mean': stats[subject][video]['video']['mean'],
                'video_var': stats[subject][video]['video']['var']
            })
    
    return pd.DataFrame(data)


def identify_outliers(df, x_col, y_col):
    """
    Identify outliers in boxplots based on the 1.5 * IQR rule.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        x_col (str): Column name for x-axis (grouping variable)
        y_col (str): Column name for y-axis (metric to analyze)
        
    Returns:
        dict: Dictionary with outliers for each group
    """
    outliers = {}
    for group in df[x_col].unique():
        subset = df[df[x_col] == group]
        q1 = subset[y_col].quantile(0.25)
        q3 = subset[y_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers for this group
        group_outliers = subset[(subset[y_col] < lower_bound) | (subset[y_col] > upper_bound)]
        if not group_outliers.empty:
            outliers[group] = group_outliers[['subject', y_col]].values.tolist()
    
    return outliers


def print_outliers(outliers_dict, metric_name):
    """
    Print outliers in a readable format.
    
    Args:
        outliers_dict (dict): Dictionary with outliers
        metric_name (str): Name of the metric being analyzed
    """
    print(f"\n{metric_name} Outliers:")
    if not outliers_dict:
        print("No outliers found")
        return
    
    for group, outliers in outliers_dict.items():
        print(f"\n{group}:")
        for outlier in outliers:
            subject, value = outlier
            print(f"  - {subject}: {value:.6f}")


def create_fmri_boxplots(df, save_path=None, figsize=(16, 20)):
    """
    Create boxplots for fMRI statistics.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
        figsize (tuple, optional): Figure size as (width, height)
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig = plt.figure(figsize=figsize)
    
    # 1. Box plot for fMRI mean values
    plt.subplot(4, 1, 1)
    sns.boxplot(x='video', y='fmri_mean', data=df)
    plt.title('Mean fMRI Values by Video (Across All Subjects)', fontsize=14)
    plt.xlabel('Video', fontsize=12)
    plt.ylabel('Mean', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Box plot for fMRI variance
    plt.subplot(4, 1, 2)
    sns.boxplot(x='video', y='fmri_var', data=df)
    plt.title('Variance of fMRI Values by Video (Across All Subjects)', fontsize=14)
    plt.xlabel('Video', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Box plot for fMRI minimum values
    plt.subplot(4, 1, 3)
    sns.boxplot(x='video', y='fmri_min', data=df)
    plt.title('Minimum fMRI Values by Video (Across All Subjects)', fontsize=14)
    plt.xlabel('Video', fontsize=12)
    plt.ylabel('Minimum', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Box plot for fMRI maximum values
    plt.subplot(4, 1, 4)
    sns.boxplot(x='video', y='fmri_max', data=df)
    plt.title('Maximum fMRI Values by Video (Across All Subjects)', fontsize=14)
    plt.xlabel('Video', fontsize=12)
    plt.ylabel('Maximum', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_summary_statistics(df, save_path=None):
    """
    Generate summary statistics by video and optionally save to CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        save_path (str, optional): Path to save the summary CSV. If None, summary is not saved.
        
    Returns:
        pandas.DataFrame: Summary statistics
    """
    summary = df.groupby('video').agg({
        'fmri_mean': ['mean', 'std', 'min', 'max'],
        'fmri_var': ['mean', 'std', 'min', 'max'],
        'fmri_min': ['mean', 'std', 'min', 'max'],
        'fmri_max': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    if save_path:
        summary.to_csv(save_path)
        print(f"Summary statistics saved to '{save_path}'")
    
    return summary


def analyze_fmri_data(pickle_path, save_plots=True, save_summary=True):
    """
    Main function to analyze fMRI data - combines all steps.
    
    Args:
        pickle_path (str): Path to the pickle file
        save_plots (bool): Whether to save the plots
        save_summary (bool): Whether to save the summary statistics
        
    Returns:
        tuple: (DataFrame, summary, outliers_dict)
    """
    # Load and process data
    df = load_and_process_data(pickle_path)
    
    # Create boxplots
    if save_plots:
        plot_path = 'fmri_statistics_boxplots.png'
    else:
        plot_path = None
    fig = create_fmri_boxplots(df, save_path=plot_path)
    
    # Generate summary statistics
    if save_summary:
        summary_path = 'fmri_statistics_summary_by_video.csv'
    else:
        summary_path = None
    summary = generate_summary_statistics(df, save_path=summary_path)
    
    # Identify outliers
    outliers = {
        'mean': identify_outliers(df, 'video', 'fmri_mean'),
        'var': identify_outliers(df, 'video', 'fmri_var'),
        'min': identify_outliers(df, 'video', 'fmri_min'),
        'max': identify_outliers(df, 'video', 'fmri_max')
    }
    
    return df, summary, outliers
'''

