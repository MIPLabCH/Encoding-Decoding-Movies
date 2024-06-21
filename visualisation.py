# -*- coding: utf-8 -*-

"""
This file contains functions related to visualisation of the model results.

Functions:
    plot_train_losses
    plot_metrics
    plot_decoder_predictions
    one_sample_permutation_test
    compute_permutation
    significant_voxels
    two_samples_permutation_test
    diff_means
"""

from imports import np, torch, F, plt, sns, explained_variance_score, stats, Pool, perm_test, nib
from dataset import normalize


### TRAINING VISUALISATION ###


def plot_train_losses(history, start_epoch):
    """
    Plot the training losses and other metrics from the training history.

    Parameters:
        history (dict): Contains the training history data including loss and other metrics.
        start_epoch (int): The starting epoch number for plotting.

    Output:
        Displays plots for each metric in the history.
    """
    x = np.arange(len(history['total_loss'])) + start_epoch 
    
    plt.plot(x, history['total_loss'])
    plt.title(history['metrics_names'][-1])
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.show()

    for i in range(len(history['metrics_names']) - 1):
        plt.plot(x, history['other_metrics'][:, i])
        plt.title(history['metrics_names'][i])
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.show()


### TESTING VISUALISATION ###


def plot_metrics(label, prediction, movie, plot_TR = False):
    """
    Plot various statistical metrics and distributions for the prediction quality of a model.

    Parameters:
        label (np.array): Ground truth values, shape (n_samples, n_features).
        prediction (np.array): Model predictions, shape (n_samples, n_features).
        movie (str): Name of the dataset or context for the plot titles.
        plot_TR (bool, optional): If true, also plots time-resolved (TR) metrics. Default is False.

    Output:
        Visualizations of correlation, explained variance, and other statistical measures between label and prediction.
    """

    TR, voxels = prediction.shape

    #z-score for visualisation
    label = (label - np.mean(label, axis=0)) / np.std(label, axis=0)
    prediction = (prediction - np.mean(prediction, axis=0)) / np.std(prediction, axis=0)

    print(f'### {movie}: ({TR}, {voxels}) ###')

    correlations_voxel = np.array([
        stats.pearsonr(label[:, i], prediction[:, i])[0] 
        for i in range(voxels) 
        if np.std(label[:, i]) != 0 and np.std(prediction[:, i]) != 0
    ])
    mean_correlation_voxel = np.mean(correlations_voxel)
    median_correlation_voxel = np.median(correlations_voxel)
    std_correlation_voxel = np.std(correlations_voxel)

    best_voxel = np.argmax(correlations_voxel)
    worst_voxel = np.argmin(correlations_voxel)

    ev_voxel = np.array([
        explained_variance_score(label[:, i], prediction[:, i]) 
        for i in range(voxels) 
        if np.std(label[:, i]) != 0
    ])
    mean_ev_voxel = np.mean(ev_voxel)
    median_ev_voxel = np.median(ev_voxel)
    std_ev_voxel = np.std(ev_voxel)

    mse_voxel = np.array([
        F.mse_loss(torch.from_numpy(label[:, i]).unsqueeze(0), torch.from_numpy(prediction[:, i]).unsqueeze(0), reduction='mean').item()
        for i in range(voxels)
        if np.std(label[:, i]) != 0  
    ])
    mean_mse_voxel = np.mean(mse_voxel)
    median_mse_voxel = np.median(mse_voxel)
    std_mse_voxel = np.std(mse_voxel)
    
    cosine_voxel = np.array([
        1 - F.cosine_similarity(torch.from_numpy(label[:, i]).unsqueeze(0), torch.from_numpy(prediction[:, i]).unsqueeze(0)).item()
        for i in range(voxels)
        if np.std(label[:, i]) != 0  # Use only voxels with non-zero variance
    ])
    mean_cosine_voxel = np.mean(cosine_voxel)
    median_cosine_voxel = np.median(cosine_voxel)
    std_cosine_voxel = np.std(cosine_voxel)

    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    plt.hist(correlations_voxel, bins=30, alpha=0.5, color='green', density=True, label='Histogram')
    sns.kdeplot(correlations_voxel, fill=True, color='g', label='KDE')
    plt.axvline(x=median_correlation_voxel, color='r', linestyle=':', linewidth=1, label=f'Median: {median_correlation_voxel:.3f}')
    plt.axvline(x=mean_correlation_voxel, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_correlation_voxel:.3f}')
    plt.axvspan(mean_correlation_voxel - std_correlation_voxel, mean_correlation_voxel + std_correlation_voxel, color='red', alpha=0.05, label=f'std: {std_correlation_voxel:.3f}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Density')
    plt.title(f'Distribution of Correlations per Voxel (N = {voxels})')
    plt.legend() 
    plt.subplot(222)
    plt.hist(ev_voxel, bins=30, alpha=0.5, color='blue', density=True, label='Histogram')
    sns.kdeplot(ev_voxel, fill=True, color='b', label='KDE')
    plt.axvline(x=median_ev_voxel, color='r', linestyle=':', linewidth=1, label=f'Median: {median_ev_voxel:.3f}')
    plt.axvline(x=mean_ev_voxel, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_ev_voxel:.3f}')
    plt.axvspan(mean_ev_voxel - std_ev_voxel, mean_ev_voxel + std_ev_voxel, color='red', alpha=0.05, label=f'std: {std_ev_voxel:.3f}')
    plt.xlabel('Explained Variance Score')
    plt.ylabel('Density')
    plt.title(f'Distribution of Explained Variances per Voxel (N = {voxels})')
    plt.legend() 
    plt.subplot(223)
    plt.hist(mse_voxel, bins=30, alpha=0.5, color='red', density=True, label='Histogram')
    sns.kdeplot(mse_voxel, fill=True, color='r', label='KDE')
    plt.axvline(x=median_mse_voxel, color='r', linestyle=':', linewidth=1, label=f'Median: {median_mse_voxel:.3f}')
    plt.axvline(x=mean_mse_voxel, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_mse_voxel:.3f}')
    plt.axvspan(mean_mse_voxel - std_mse_voxel, mean_mse_voxel + std_mse_voxel, color='red', alpha=0.05, label=f'std: {std_mse_voxel:.3f}')
    plt.xlabel('MSE')
    plt.ylabel('Density')
    plt.title(f'Distribution of MSE per Voxel (N = {voxels})')
    plt.legend() 
    plt.subplot(224)
    plt.hist(cosine_voxel, bins=30, alpha=0.5, color='purple', density=True, label='Histogram')
    sns.kdeplot(cosine_voxel, fill=True, color='purple', label='KDE')
    plt.axvline(x=median_cosine_voxel, color='r', linestyle=':', linewidth=1, label=f'Median: {median_cosine_voxel:.3f}')
    plt.axvline(x=mean_cosine_voxel, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_cosine_voxel:.3f}')
    plt.axvspan(mean_cosine_voxel - std_cosine_voxel, mean_cosine_voxel + std_cosine_voxel, color='red', alpha=0.05, label=f'std: {std_cosine_voxel:.3f}')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Density')
    plt.title(f'Distribution of Cosine Distances per Voxel (N = {voxels})')
    plt.legend() 
    plt.show()

    plt.figure(figsize=(20, 7))
    plt.subplot(2, 1, 1)
    plt.title(f'Best Voxel (Corr({best_voxel}) = {np.max(correlations_voxel):.3f})')
    plt.plot(label[:,best_voxel], label = 'ground truth')
    plt.plot(prediction[:,best_voxel], label = 'prediction')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title(f'Worst Voxel (Corr({worst_voxel}) = {np.min(correlations_voxel):.3f})')
    plt.plot(label[:,worst_voxel], label = 'ground truth')
    plt.plot(prediction[:,worst_voxel], label = 'prediction')
    plt.legend()
    plt.show()

    if plot_TR:
    
        correlations_TR = np.array([
            stats.pearsonr(label[i, :], prediction[i, :])[0] 
            for i in range(TR) 
            if np.std(label[i, :]) != 0 and np.std(prediction[i, :]) != 0
        ])
        mean_correlation_TR = np.mean(correlations_TR)
        median_correlation_TR = np.median(correlations_TR)
        std_correlation_TR = np.std(correlations_TR)
    
        best_TR = np.argmax(correlations_TR)
        worst_TR = np.argmin(correlations_TR)
    
        ev_TR = np.array([
            explained_variance_score(label[i, :], prediction[i, :]) 
            for i in range(TR) 
            if np.std(label[i, :]) != 0
        ])
        mean_ev_TR = np.mean(ev_TR)
        median_ev_TR = np.median(ev_TR)
        std_ev_TR = np.std(ev_TR)
    
        mse_TR = np.array([
            F.mse_loss(torch.from_numpy(label[i, :]).unsqueeze(0), torch.from_numpy(prediction[i, :]).unsqueeze(0), reduction='mean').item()
            for i in range(TR)
            if np.std(label[i, :]) != 0  
        ])
        mean_mse_TR = np.mean(mse_TR)
        median_mse_TR = np.median(mse_TR)
        std_mse_TR = np.std(mse_TR)
    
        cosine_TR = np.array([
            1 - F.cosine_similarity(torch.from_numpy(label[i, :]).unsqueeze(0), torch.from_numpy(prediction[i, :]).unsqueeze(0)).item()
            for i in range(TR)
            if np.std(label[i, :]) != 0  # Use only voxels with non-zero variance
        ])
        mean_cosine_TR = np.mean(cosine_TR)
        median_cosine_TR = np.median(cosine_TR)
        std_cosine_TR = np.std(cosine_TR)
    
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.hist(correlations_TR, bins=30, alpha=0.5, color='green', density=True, label='Histogram')
        sns.kdeplot(correlations_TR, fill=True, color='g', label='KDE')
        plt.axvline(x=median_correlation_TR, color='r', linestyle=':', linewidth=1, label=f'Median: {median_correlation_TR:.3f}')
        plt.axvline(x=mean_correlation_TR, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_correlation_TR:.3f}')
        plt.axvspan(mean_correlation_TR - std_correlation_TR, mean_correlation_TR + std_correlation_TR, color='red', alpha=0.05, label=f'std: {std_correlation_TR:.3f}')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Density')
        plt.title(f'Distribution of Correlations per TR (N = {TR})')
        plt.legend() 
        plt.subplot(222)
        plt.hist(ev_TR, bins=30, alpha=0.5, color='blue', density=True, label='Histogram')
        sns.kdeplot(ev_TR, fill=True, color='b', label='KDE')
        plt.axvline(x=median_ev_TR, color='r', linestyle=':', linewidth=1, label=f'Median: {median_ev_TR:.3f}')
        plt.axvline(x=mean_ev_TR, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_ev_TR:.3f}')
        plt.axvspan(mean_ev_TR - std_ev_TR, mean_ev_TR + std_ev_TR, color='red', alpha=0.05, label=f'std: {std_ev_TR:.3f}')
        plt.xlabel('Explained Variance Score')
        plt.ylabel('Density')
        plt.title(f'Distribution of Explained Variances per TR (N = {TR})')
        plt.legend()  
        plt.subplot(223)
        plt.hist(mse_TR, bins=30, alpha=0.5, color='red', density=True, label='Histogram')
        sns.kdeplot(mse_TR, fill=True, color='r', label='KDE')
        plt.axvline(x=median_mse_TR, color='r', linestyle=':', linewidth=1, label=f'Median: {median_mse_TR:.3f}')
        plt.axvline(x=mean_mse_TR, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_mse_TR:.3f}')
        plt.axvspan(mean_mse_TR - std_mse_TR, mean_mse_TR + std_mse_TR, color='red', alpha=0.05, label=f'std: {std_mse_TR:.3f}')
        plt.xlabel('MSE')
        plt.ylabel('Density')
        plt.title(f'Distribution of MSE per TR (N = {TR})')
        plt.legend() 
        plt.subplot(224)
        plt.hist(cosine_TR, bins=30, alpha=0.5, color='purple', density=True, label='Histogram')
        sns.kdeplot(cosine_TR, fill=True, color='purple', label='KDE')
        plt.axvline(x=median_cosine_TR, color='r', linestyle=':', linewidth=1, label=f'Median: {median_cosine_TR:.3f}')
        plt.axvline(x=mean_cosine_TR, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_cosine_TR:.3f}')
        plt.axvspan(mean_cosine_TR - std_cosine_TR, mean_cosine_TR + std_cosine_TR, color='red', alpha=0.05, label=f'std: {std_cosine_TR:.3f}')
        plt.xlabel('Cosine Distance')
        plt.ylabel('Density')
        plt.title(f'Distribution of Cosine Distances per TR (N = {TR})')
        plt.legend() 
        plt.show()
    
        plt.figure(figsize=(20, 7))
        plt.subplot(2, 1, 1)
        plt.title(f'Best TR (Corr({best_TR}) = {np.max(correlations_TR):.3f})')
        plt.plot(label[best_TR, :], label = 'ground truth')
        plt.plot(prediction[best_TR, :], label = 'prediction')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.title(f'Worst TR (Corr({worst_TR}) = {np.min(correlations_TR):.3f})')
        plt.plot(label[worst_TR, :], label = 'ground truth')
        plt.plot(prediction[worst_TR, :], label = 'prediction')
        plt.legend()
        plt.show()

def plot_decoder_predictions(predictions, videos):
    """
    Display comparison plots between original videos and their corresponding predictions.

    Parameters:
        predictions (dict): A dictionary containing predicted frames, indexed by video names.
        videos (dict): A dictionary containing original video frames, indexed by video names.

    This function iterates through each video key, retrieves frames from both the original and predicted data,
    and plots them side by side for visual comparison. It supports different intervals for videos with a large
    number of frames, adjusting the subplot layout accordingly.
    """
    for key in videos.keys():
        prediction = predictions[key]
        video = videos[key][..., 15]
        print(f'## {key} ({video.shape[0]}) ##')
        if prediction.shape[0] > 400:
            indices = np.arange(0, prediction.shape[0], 15)
        else:
            indices = np.arange(0, prediction.shape[0], 3)
        print(f'Display {len(indices)} frames:', indices)
        col = int(indices.shape[0] / 3) + 1
        k = 0

        if len(indices) < 20:
            plt.figure(figsize=(8, 10))
        elif len(indices) >= 20 and len(indices) < 40:
            plt.figure(figsize=(10, 15))
        else:
            plt.figure(figsize=(15, 20))
            
        for i in indices:
            k += 1
            plt.subplot(col,6,k)
            plt.imshow(np.transpose(normalize(video[i]), (1, 2, 0)))
            plt.axis('off')
            k += 1
            plt.subplot(col,6,k)
            plt.imshow(np.transpose(normalize(prediction[i]), (1, 2, 0)))
            plt.axis('off')
        plt.show()
        del prediction, video, indices


### STATISTICAL TESTS ###

# Define global variables to hold data, assuming they fit comfortably in memory
# and can be accessed by subprocesses in multiprocessing environment.
global tot_lab, tot_pred, valid_indices

def one_sample_permutation_test(lab, pred, alpha = 0.05, num_permutations = 999):
    """
    Conducts a one-sample permutation test to evaluate the significance of observed correlations against a null distribution.

    Parameters:
        lab (numpy array): The ground truth labels, expected to be of shape (TR, voxels).
        pred (numpy array): The model predictions, expected to be of the same shape as 'lab'.
        alpha (float): Significance level for determining statistical significance.
        num_permutations (int): Number of permutations to generate the null distribution.

    Returns:
        numpy array: P-values for each voxel, providing insight into the significance of the observed correlations.
    
    This function computes the correlations between predicted and actual data for each voxel and compares these
    correlations against a null distribution generated by randomly shuffling the labels.
    """
    tot_lab, tot_pred = lab, pred

    TR, voxels = tot_pred.shape
    
    # Calculate valid indices only once
    valid_indices = [i for i in range(voxels) if np.std(tot_lab[:, i]) != 0 and np.std(tot_pred[:, i]) != 0]
    
    # Compute observed correlations using valid indices
    correlations_voxel = np.array([
        stats.pearsonr(tot_lab[:, i], tot_pred[:, i])[0] 
        for i in valid_indices
    ])
    mean_correlation_voxel = np.mean(correlations_voxel)
    median_correlation_voxel = np.median(correlations_voxel)
    std_correlation_voxel = np.std(correlations_voxel)

    # Using multiprocessing to handle the computationally intensive task
    with Pool() as pool:
        # Create unique seeds based on the process index, could be random or sequential
        seeds = np.random.randint(0, 1000000, size=num_permutations)
        correlations_voxel_perm = np.asarray(pool.map(compute_permutation, seeds)).T
    mean_correlations_perm = np.concatenate(correlations_voxel_perm[:, :10], axis=0) #for visualisation of the null distribution

    plt.figure(figsize=(8, 4))
    plt.hist(correlations_voxel, bins=30, alpha=0.5, color='green', density=True, label='Prediction Distribution')
    sns.kdeplot(correlations_voxel, fill=True, color='g')
    plt.axvline(x=median_correlation_voxel, color='r', linestyle=':', linewidth=1, label=f'Median: {median_correlation_voxel:.3f}')
    plt.axvline(x=mean_correlation_voxel, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_correlation_voxel:.3f}')
    plt.axvspan(mean_correlation_voxel - std_correlation_voxel, mean_correlation_voxel + std_correlation_voxel, color='red', alpha=0.05, label=f'std: {std_correlation_voxel:.3f}')
    
    # Plotting the histogram for mean_correlations_perm
    plt.hist(mean_correlations_perm, bins=100, alpha=0.5, color='grey', density=True, label=f'Null Distribution\n(Permutation Test)')
    sns.kdeplot(mean_correlations_perm, fill=True, color='grey')

    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Density')
    plt.title(f'Distribution of Correlations per Voxel (N = {voxels})')
    plt.legend()
    plt.show()
    
    # Calculate the p-values for each observed correlation
    p_values = np.sum(correlations_voxel[:, np.newaxis] < correlations_voxel_perm, axis=1)/num_permutations
    # Assuming you have an array of p-values called `p_values` and other variables `num_permutations`, `voxels`
    plt.figure(figsize=(8, 2))
    # Add a very small value to p_values to avoid log(0) issue, assuming no p-value is exactly 0
    adjusted_p_values = np.maximum(p_values, 1e-6)
    plt.hist(adjusted_p_values, bins=1000, alpha=0.2, color='b', cumulative=True, density=False)
    plt.axvline(alpha, color='k', linestyle=':', linewidth=0.6)
    plt.text(alpha, 170, f' alpha = {alpha}', color='k', ha='left')
    plt.axhline(y=np.sum(p_values < alpha), color='k', linestyle=':', linewidth=0.6)
    plt.text(alpha, 170+np.sum(p_values < alpha), f' {100*np.sum(p_values < alpha)/voxels:.1f}%', color='r', ha='left')
    plt.plot(alpha, np.sum(p_values < alpha), 'x', markersize=6, markeredgewidth=2, color='red')
    plt.axvline(alpha/voxels, color='k', linestyle=':', linewidth=0.6)
    plt.text(alpha/voxels, 170, f' alpha = {alpha}/{voxels}', color='k', ha='left')
    plt.axhline(y=np.sum(p_values < alpha/voxels), color='k', linestyle=':', linewidth=0.6)
    plt.text(alpha/voxels, 170+np.sum(p_values < alpha/voxels), f' {100*np.sum(p_values < alpha/voxels)/voxels:.1f}%', color='r', ha='left')
    plt.plot(alpha/voxels, np.sum(p_values < alpha/voxels), 'x', markersize=6, markeredgewidth=2, color='red')
    plt.xscale('log')
    plt.xlabel('p-value (log scale)')
    plt.ylabel('Cumulative count')
    plt.xlim([0.000001, 1.001])
    plt.title(f'Cumulative Distribution of p-values (Permutation Test with {num_permutations} permutations)')
    plt.show()
    return p_values

# Define the compute_permutation function at the top level
def compute_permutation(seed):
    """
    Compute a permutation of Pearson correlation coefficients between shuffled labels and actual predictions.

    Parameters:
        seed (int): A seed number used to ensure reproducibility of the random shuffling process.

    Returns:
        numpy array: An array of Pearson correlation coefficients calculated between shuffled labels
                     and actual predictions for each valid index.

    This function shuffles the labels array using a given seed to ensure the randomness is reproducible, 
    then calculates the Pearson correlation coefficient between these shuffled labels and the original predictions.
    It operates on the global variables `tot_lab`, `tot_pred`, and `valid_indices`, which must be defined outside
    this function. These variables represent the total labels, total predictions, and indices of valid data points, 
    respectively.
    """
    np.random.seed(seed)  # Set a unique seed for each process
    shuffled_labels = np.copy(tot_lab)
    np.random.shuffle(shuffled_labels)  # Shuffle labels
    perm = np.array([
        stats.pearsonr(shuffled_labels[:, i], tot_pred[:, i])[0] 
        for i in valid_indices
    ])
    return perm

def significant_voxels(p_values, alpha = 0.05, voxels = 4330):
    """
    Identify and visualize (spacially) significant voxels from fMRI data based on provided p-values.

    Parameters:
        p_values (numpy array): Array of p-values for each voxel.
        alpha (float): Significance level for the testing, default is 0.05.
        voxels (int): Total number of voxels considered in the analysis, default is 4330.
    """
    # you can choose another fMRI data as template
    brain_path = '/media/miplab-nas2/Data2/Movies_Emo/Preprocessed_data/sub-S01/ses-1/pp_sub-S01_ses-1_YouAgain.feat/filtered_func_data_res_MNI.nii'
    brain_img = nib.load(brain_path)
    brain_data = np.asanyarray(brain_img.dataobj)[..., 50] #only look at 1 TR

    # path to the mask
    mask_path = '/home/chchan/Michael-Nas2/ml-students2023/resources/vis_mask.nii'
    img_mask = nib.load(mask_path)
    mask_2d = np.asanyarray(img_mask.dataobj).reshape(-1,)
    indices = np.where(mask_2d == 1)[0] # 4330 relevant voxels
    
    #zero-filled array with the same shape as the original 3D data, and flatten
    significant_voxels = np.zeros(brain_data.shape, dtype=np.float32).flatten()
    #set to 1 the voxels that have a p-value < alpha/voxels
    significant_voxels[indices] = np.where(p_values > alpha/voxels, 0, 1)
    #reshape back to the original 3D shape
    significant_voxels = significant_voxels.reshape(brain_data.shape)

    #plot some brain slices
    for i in range(0, 90, 5):
        plt.subplot(1,3,1)
        plt.imshow(np.maximum(significant_voxels[i,:,:],normalize(brain_data[i,:,:])), cmap='gray')
        plt.subplot(1,3,2)
        plt.imshow(np.maximum(significant_voxels[:,i,:],normalize(brain_data[:,i,:])), cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow(np.maximum(significant_voxels[:,:,i],normalize(brain_data[:,:,i])), cmap='gray')
        plt.show()


def two_samples_permutation_test(model1, lab1, pred1, model2, lab2, pred2, num_permutations = 999):
    """
    Conducts a two-sample permutation test to compare two sets of correlations between predicted and actual data.

    Parameters:
        model1 (str): Name or identifier for the first model.
        lab1 (numpy array): Ground truth labels for the first model.
        pred1 (numpy array): Predictions from the first model.
        model2 (str): Name or identifier for the second model.
        lab2 (numpy array): Ground truth labels for the second model.
        pred2 (numpy array): Predictions from the second model.
        num_permutations (int): Number of permutations to perform.

    Returns:
        Displays a visual comparison of correlation distributions and reports the p-value from the permutation test.
    
    This function compares the mean difference in correlations between two different models or conditions using a
    permutation test, providing a statistical comparison of their performance.
    """
    all_encoded, all_labels = [], []
    for key in pred1.keys():
        all_encoded.append(pred1[key])
        all_labels.append(lab1[key])
    tot_pred = np.concatenate(all_encoded, axis=0)
    tot_lab = np.concatenate(all_labels, axis=0)
    voxels = tot_pred.shape[1]
    valid_indices = [i for i in range(voxels) if np.std(tot_lab[:, i]) != 0 and np.std(tot_pred[:, i]) != 0]
    sample1 = np.array([
        stats.pearsonr(tot_lab[:, i], tot_pred[:, i])[0] 
        for i in valid_indices
    ])
    mean1= np.mean(sample1)

    all_encoded, all_labels = [], []
    for key in pred2.keys():
        all_encoded.append(pred2[key])
        all_labels.append(lab2[key])
    tot_pred = np.concatenate(all_encoded, axis=0)
    tot_lab = np.concatenate(all_labels, axis=0)
    voxels = tot_pred.shape[1]
    valid_indices = [i for i in range(voxels) if np.std(tot_lab[:, i]) != 0 and np.std(tot_pred[:, i]) != 0]
    sample2 = np.array([
        stats.pearsonr(tot_lab[:, i], tot_pred[:, i])[0] 
        for i in valid_indices
    ])
    mean2= np.mean(sample2)
    
    test_result = perm_test((sample1, sample2), diff_means, permutation_type='independent', alternative='less', n_resamples=num_permutations)

    plt.figure(figsize=(8, 4))
    plt.hist(sample1, bins=30, alpha=0.5, color='salmon', density=True)
    sns.kdeplot(sample1, fill=True, color='red', label=model1)
    plt.axvline(mean1, color='r', linestyle='--', linewidth=1)
    plt.text(mean1, 8.8, f'Mean: {mean1:.3f} ', color='r', ha='right')
    plt.hist(sample2, bins=30, alpha=0.5, color='lightgreen', density=True)
    sns.kdeplot(sample2, fill=True, color='g', label=model2)
    plt.axvline(mean2, color='g', linestyle='--', linewidth=1)
    plt.text(mean2, 8.8, f' Mean: {mean2:.3f}', color='g', ha='left')
    plt.title(f'Two samples permutation test ({num_permutations} permutations)\nMean difference: {test_result.statistic} (p = {test_result.pvalue:.3e})')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Define the statistic function
def diff_means(x, y):
    return np.mean(x) - np.mean(y)

