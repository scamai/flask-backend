import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

def transform_image(image_path, transform):
    image = cv2.imread(image_path)
    # Convert BGR image to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0)
    return image

def get_datasetname_with_dirs():
    CELEB_test_set_real = '/deepfake/dataset/CELEB_test_set/real'
    CELEB_test_set_fake = '/deepfake/dataset/CELEB_test_set/fake'
    CELEB_real = '/deepfake/dataset/CELEB-20240704T223847Z-001/test/real'
    CELEB_fake = '/deepfake/dataset/CELEB-20240704T223847Z-001/test/fake'
    CELEB_M_fake = '/deepfake/dataset/CELEB-M-20240711T043947Z-001/test/fake'
    CELEB_M_real = '/deepfake/dataset/CELEB-M-20240711T043947Z-001/test/real'
    DF_real = '/deepfake/dataset/DF-20240711T045921Z-001/test/real'
    DF_fake = '/deepfake/dataset/DF-20240711T045921Z-001/test/fake'
    DFD_real = '/deepfake/dataset/DFD-20240711T043951Z-001/test/real'
    DFD_fake = '/deepfake/dataset/DFD-20240711T043951Z-001/test/fake'
    F2F_real = '/deepfake/dataset/F2F-20240711T050229Z-001/test/real'
    F2F_fake = '/deepfake/dataset/F2F-20240711T050229Z-001/test/fake'
    FS_real = '/deepfake/dataset/FS_test_set/real'
    FS_fake = '/deepfake/dataset/FS_test_set/fake'
    FS_I_real = '/deepfake/dataset/FS-I-20240711T043958Z-001/test/real'
    FS_I_fake = '/deepfake/dataset/FS-I-20240711T043958Z-001/test/fake'
    NT_I_real = '/deepfake/dataset/NT_I_test/real'
    NT_I_fake = '/deepfake/dataset/NT_I_test/fake'
    NT_or_SFNT_real = '/deepfake/dataset/NT_or_SFNT/real'
    NT_or_SFNT_fake = '/deepfake/dataset/NT_or_SFNT/fake'
    kaggle_real = '/deepfake/dataset/kaggle_/evaluation_set_1k/real'
    kaggle_fake = '/deepfake/dataset/kaggle_/evaluation_set_1k/fake'

    inputs = [('CELEB', CELEB_real, CELEB_fake),
            ('CELEB_M', CELEB_M_real, CELEB_M_fake),
            ('DF', DF_real, DF_fake),
            ('DFD', DFD_real, DFD_fake),
            ('F2F', F2F_real, F2F_fake),
            ('FS', FS_real, FS_fake),
            ('FS_I', FS_I_real, FS_I_fake),
            ('NT_I', NT_I_real, NT_I_fake),
            ('NT_or_SFNT', NT_or_SFNT_real, NT_or_SFNT_fake),
            ('kaggle', kaggle_real, kaggle_fake)]
    return inputs

def get_list_of_evaluation_dataset_paths(only_last_n_dataset=-1):
    """
    param: only_last_n_dataset: This controls we only take the last n folders listed, so we dont run duplicated evaluations (when we add new datasets)
    """
    CELEB_test_set_real = '/deepfake/dataset/CELEB_test_set/real'
    CELEB_test_set_fake = '/deepfake/dataset/CELEB_test_set/fake'
    CELEB_real = '/deepfake/dataset/CELEB-20240704T223847Z-001/test/real'
    CELEB_fake = '/deepfake/dataset/CELEB-20240704T223847Z-001/test/fake'
    CELEB_M_fake = '/deepfake/dataset/CELEB-M-20240711T043947Z-001/test/fake'
    CELEB_M_real = '/deepfake/dataset/CELEB-M-20240711T043947Z-001/test/real'
    DF_real = '/deepfake/dataset/DF-20240711T045921Z-001/test/real'
    DF_fake = '/deepfake/dataset/DF-20240711T045921Z-001/test/fake'
    DFD_real = '/deepfake/dataset/DFD-20240711T043951Z-001/test/real'
    DFD_fake = '/deepfake/dataset/DFD-20240711T043951Z-001/test/fake'
    F2F_real = '/deepfake/dataset/F2F-20240711T050229Z-001/test/real'
    F2F_fake = '/deepfake/dataset/F2F-20240711T050229Z-001/test/fake'
    FS_real = '/deepfake/dataset/FS_test_set/real'
    FS_fake = '/deepfake/dataset/FS_test_set/fake'
    FS_I_real = '/deepfake/dataset/FS-I-20240711T043958Z-001/test/real'
    FS_I_fake = '/deepfake/dataset/FS-I-20240711T043958Z-001/test/fake'
    NT_I_real = '/deepfake/dataset/NT_I_test/real'
    NT_I_fake = '/deepfake/dataset/NT_I_test/fake'
    NT_or_SFNT_real = '/deepfake/dataset/NT_or_SFNT/real'
    NT_or_SFNT_fake = '/deepfake/dataset/NT_or_SFNT/fake'
    kaggle_real = '/deepfake/dataset/kaggle_/evaluation_set_1k/real'
    kaggle_fake = '/deepfake/dataset/kaggle_/evaluation_set_1k/fake'

    dataset_real_paths_str_list = ['CELEB_test_set_real', 'CELEB_M_real', 'DF_real', 
                               'DFD_real', 'F2F_real', 'FS_real', 'FS_I_real', 
                               'NT_I_real', 'NT_or_SFNT_real', 'kaggle_real']
    if only_last_n_dataset > 0:
        # Evaluating on only the new datasets
        dataset_real_paths_str_list = dataset_real_paths_str_list[-only_last_n_dataset:]
        
    dataset_real_paths_list = [eval(path_str) for path_str in dataset_real_paths_str_list]
    dataset_fake_paths_list = [eval(path_str.replace('real','fake')) for path_str in dataset_real_paths_str_list]
    return dataset_real_paths_str_list, dataset_real_paths_list, dataset_fake_paths_list

def get_image_paths_and_target(only_last_n_dataset=-1):
    _, dataset_real_paths_list, dataset_fake_paths_list = get_list_of_evaluation_dataset_paths(only_last_n_dataset=only_last_n_dataset)
    target = []
    real_image_paths = []
    for directory in dataset_real_paths_list:
        current_real_image_paths = _get_image_paths(directory)
        real_image_paths.extend(current_real_image_paths)
        target.extend([0]*len(current_real_image_paths))

    fake_image_paths = []
    for directory in dataset_fake_paths_list:
        current_fake_image_paths = _get_image_paths(directory)
        fake_image_paths.extend(current_fake_image_paths)
        target.extend([1]*len(current_fake_image_paths))

    return real_image_paths, fake_image_paths, target

def _get_image_paths(directory):
    file_names = os.listdir(directory)
    image_paths = []
    for file_name in file_names:
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            image_paths.append(os.path.join(directory, file_name))
    return image_paths


class EvaluationDataset(Dataset):
    def __init__(self, transform, norm_255, AutoImageProcessor, only_last_n_dataset):
        # First we get the images and paths
        real_image_paths, fake_image_paths, target = get_image_paths_and_target(only_last_n_dataset)
        self.img_paths = real_image_paths + fake_image_paths
        self.img_labels = target
        self.transform = transform
        self.norm_255 = norm_255
        self.AutoImageProcessor = AutoImageProcessor

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        # Convert BGR image to RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('double')
        label = self.img_labels[idx]

        # If this is from AutoImageProcessor (that comes with model in HF)
        if self.AutoImageProcessor:
            image = self.AutoImageProcessor(image, return_tensors="pt")
            image = image['pixel_values'][0, :, :, :]
            return image, label

        # Normalize it to 0-1 instead of 0-255
        if self.norm_255:
            assert np.max(image) <= 255, 'max = {} is out of bound, not within 255 in pixel value!!!'.format(np.max(image))

            # Normalize if this is larger than 125 to 0-1
            if np.max(image) > 125:
                image /= 255.0 # Normalize to 0-1 
                assert np.max(image) <= 1, 'your image is out of bound after normalization, not within 1 in pixel value!!!'

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        
        return image, label

def getEvalDataLoader(transform=None, batch_size=64, num_workers=16, norm_255=True,
                     AutoImageProcessor=None, only_last_n_dataset=-1):
    dataset = EvaluationDataset(transform=transform, norm_255=norm_255, 
                                AutoImageProcessor=AutoImageProcessor,
                                only_last_n_dataset=only_last_n_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader


def save_prediction_csv(pred_list, target_list, method_name, 
                        save_dir='/deepfake/code/agg_results/', 
                        per_dataset_mode=False,
                        dataset_name=None):
    """
    We will save the predictions into a long folder
    """
    assert len(pred_list) == len(target_list), 'Make sure the prediction list and the target list is the same length'
    # Create the folder if not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if per_dataset_mode:
        assert dataset_name is not None, 'This is per_dataset_mode for save prediciton csv, this is not yet supported yet so lets not do that :p'
        np.save(os.path.join(save_dir, '{}_{}_pred_list_full.npy'.format(method_name, dataset_name)), pred_list)
        np.save(os.path.join(save_dir, '{}_{}_target_list_full.npy'.format(method_name, dataset_name)), target_list)
    else:
        # We save all the thing together
        np.save(os.path.join(save_dir, '{}_pred_list_full.npy'.format(method_name)), pred_list)
        np.save(os.path.join(save_dir, '{}_target_list_full.npy'.format(method_name)), target_list)

def combine_multiple_model_predictions(value_model_dict, 
                                       result_dir='/deepfake/code/agg_results/',
                                       verbose=False,
                                       no_plot=True):
    """
    This function would combine the prediction of multiple models with a value model and then
    plot the auroc curves to get the result
    param: value_model_dict dict<string, float>, key is model name that matches the prefix of the saved result, 
            value is floating point number representing the value model of such model
    param: result_dir <string>: Where we saved the evaluation results

    """
    result_files = os.listdir(result_dir)
    agg_pred_list = None # Initialize it
    # Loop over the value model dictionary
    for model_name, value in value_model_dict.items():
        pred_file = '{}_pred_list_full.npy'.format(model_name)
        
        if pred_file not in result_files: # this one is full mode evaled
            contain_name_list = [file for file in result_files if file.startswith(model_name)]
            assert len(contain_name_list) > 0, 'Result dir does not contain prefix of your files,\
check dir or your input model name! Issue found in  {}'.format(model_name)
            print('It seems like you have not yet run combine_predictions.ipynb yet, go there first')
            quit()
        pred_file = os.path.join(result_dir, pred_file)
        if verbose:
            print('For method {}, Found full evaluation result!'.format(model_name))
        target_file = pred_file.replace('pred','target')
        assert os.path.exists(target_file), 'While we found pred, your target is not present, missing {}'.format(target_file)
        # Now we read the pred and target
        pred = np.load(pred_file)
        target = np.load(target_file)
        
        # Apply a reshape step to make sure they are pure lists
        pred = np.reshape(pred, (-1, ))
        if verbose:
            print('for model {}, shape of pred {}, target {}'.format(model_name, np.shape(pred), np.shape(target)))
        
        ## Adding the predictions
        if agg_pred_list is None: # Directly assign for the first one
            agg_pred_list = value * pred 
        else: # Add it to the value model if its prediction
            agg_pred_list += value * pred
        
        if verbose:
            # Debugging part
            print('avg of target')
            print(np.average(target))
    
    # Before we go to the roc curve, lets make them all positive and between 0-1
    agg_pred_list -= np.min(agg_pred_list)
    agg_pred_list /= np.max(agg_pred_list)
    
    agg_pred_list = agg_pred_list.tolist()
    target = target.tolist()
    # Now we have the agg_pred_list, we put it into evaluation
    df = None
    df = get_per_dataset_roc(target, agg_pred_list, 
                             verbose=verbose,
                             no_plot=no_plot)
    return df, target, agg_pred_list

def get_combination_two_models(model_a_name, model_b_name,
                               step=0.05):
    full_result_df = None
    for a in np.arange(0, 1+step, step):
        model_value_dict = {model_a_name: a,
                            model_b_name: 1-a}
        df, target, agg_pred = combine_multiple_model_predictions(model_value_dict, 
                                verbose=False,  no_plot=True)
        df.rename(columns={0: '{:.2f}:{:.2f}'.format(a, 1-a),}, inplace=True)
        if full_result_df is None:
            full_result_df = df
        else:
            full_result_df = pd.concat([full_result_df, df], axis=1)
    return full_result_df

def get_combination_three_models(model_a_name, model_b_name, model_c_name,
                               step=0.05):
    full_result_df = None
    for a in np.arange(0, 1+step, step):
        for b in np.arange(0, 1-a, step):
            model_value_dict = {model_a_name: a,
                                model_b_name: b,
                                model_c_name: 1-a-b}
            df, target, agg_pred = combine_multiple_model_predictions(model_value_dict, 
                                    verbose=False,  no_plot=True)
            df.rename(columns={0: '{:.2f}:{:.2f}:{:.2f}'.format(a, b, 1-a-b),}, inplace=True)
            if full_result_df is None:
                full_result_df = df
            else:
                full_result_df = pd.concat([full_result_df, df], axis=1)
    return full_result_df

def get_combination_four_models(model_a_name, model_b_name, model_c_name, model_d_name,
                               step=0.05):
    full_result_df = None
    for a in np.arange(0, 1+step, step):
        for b in np.arange(0, 1-a, step):
            for c in np.arange(0, 1-a, step):
                model_value_dict = {model_a_name: a,
                                    model_b_name: b,
                                    model_c_name: c,
                                    model_d_name: 1-a-b-c}
                df, target, agg_pred = combine_multiple_model_predictions(model_value_dict, 
                                        verbose=False,  no_plot=True)
                df.rename(columns={0: '{:.2f}:{:.2f}:{:.2f}:{:.2f}'.format(a, b,c, 1-a-b-c),}, inplace=True)
                if full_result_df is None:
                    full_result_df = df
                else:
                    full_result_df = pd.concat([full_result_df, df], axis=1)
    return full_result_df


###### Evaluation plotting start #######
def plot_roc_curve(target, pred, title_postfix='', savedir=None, no_plot=False):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(target, pred)
    roc_auc = auc(fpr, tpr)
    if no_plot:
        return roc_auc
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for ({title_postfix})')
    plt.legend()
    if savedir is not None:
        print('Saving the roc curve to ', savedir)
        if not os.path.exists(os.path.dirname(savedir)):
            os.makedirs(os.path.dirname(savedir))
        plt.savefig(savedir)
    else:
        plt.show()
    return roc_auc

def get_per_dataset_roc(target, pred, imgs_per_dataset=2000, 
                        roc_curves_dir='./roc_curves', reversed_label=False, 
                        only_last_n_dataset=-1,
                        verbose=False,
                        no_plot=False):
    """
    The mother function that generates the roc curves for each of the datasets
    """
    dataset_real_paths_str_list, _, _ = get_list_of_evaluation_dataset_paths(only_last_n_dataset)
    assert len(target) == len(pred), 'The length of the prediction and labels should be the same'
    assert len(target) %2 == 0, 'The length of the target is not even, note we assume half true and half false'
    assert len(target) / len(dataset_real_paths_str_list) == imgs_per_dataset, \
                'The length of imgs is not matching with the dataset name list'
    
    # Setup the place to store the roc curves
    if not os.path.exists(roc_curves_dir):
        os.makedirs(roc_curves_dir)
    imgs_per_real_fake_per_dataset = imgs_per_dataset // 2
    midpoint = len(target) // 2
    roc_dict = {}
    if verbose:
        print('midpoint = {}'.format(midpoint))
        print('imgs_per_real_fake_per_dataset = {}'.format(imgs_per_real_fake_per_dataset))

    for ind, dataset_name_real in enumerate(dataset_real_paths_str_list):
        # Get the dataset name 
        dataset_name = dataset_name_real.replace('_real','')
        roc_save_name = os.path.join(roc_curves_dir, dataset_name+'.png')

        start_ind = int(ind*imgs_per_real_fake_per_dataset)
        end_ind = int((ind+1)*imgs_per_real_fake_per_dataset)
        # Get the current targets and labels with two sides, from first half and the second half
        cur_target = target[start_ind:end_ind] + \
            target[start_ind + midpoint:end_ind + midpoint] 
        cur_pred = pred[start_ind:end_ind] + \
            pred[start_ind + midpoint: end_ind + midpoint] 
        
        if verbose:
            print('start_ind is {}'.format(start_ind))
            print('end_ind is {}'.format(end_ind))
            print('shape of cur_target is {}'.format(np.shape(cur_target)))
            print('shape of cur_pred is {}'.format(np.shape(cur_pred)))
            print('cur target = ', cur_target)
            
        assert np.mean(cur_target) == 0.5, 'Your average target is not 0.5, it is {}'.format(np.mean(cur_target))
        if reversed_label: # For some models the labels are reversed
            cur_target = [1-x for x in cur_target]
        roc = plot_roc_curve(cur_target, cur_pred, title_postfix=dataset_name, 
                             savedir=roc_save_name, no_plot=no_plot)
        roc_dict[dataset_name] = roc
    
    df = pd.DataFrame.from_dict(data=roc_dict, orient='index' )
    return df