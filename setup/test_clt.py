# -*- coding: utf-8 -*-
"""
@author(s): Adam Gibicar, Samir Mitha
"""

import sys
import configparser as ConfParser
import subprocess
import shutil
import argparse
import os
import time
import numpy as np
import nibabel as nib
from models import select_model
from utils import (binarize,
                   normalize_vol,
                    get_pred_4ch_agg_patches)

def preprocess(vol_1):
    patient = os.path.dirname(vol_1)
    output = patient
    templateFlair = None
    intermediateFolder = args.intermediate_folder

    # The configuration file for anima is ~/.anima/config.txt (can be overridden with -a and -s arguments)
    configFilePath = os.path.join(os.path.expanduser("~"),'.anima', 'config.txt')

    # Open the configuration parser and exit if anima configuration cannot be loaded
    configParser = ConfParser.RawConfigParser()

    if os.path.exists(configFilePath):
        configParser.read(configFilePath)
    else:
        sys.exit('Please create a configuration file (~/.anima/config.txt) for Anima python scripts.')

    # Initialize anima directories
    animaDir = configParser.get("anima-scripts", 'anima')
    animaScriptsPublicDir = configParser.get("anima-scripts", 'anima-scripts-public-root')

    # Anima commands
    animaBrainExtraction = os.path.join(animaScriptsPublicDir, "brain_extraction", "animaAtlasBasedBrainExtraction.py")
    animaN4BiasCorrection = os.path.join(animaDir, "animaN4BiasCorrection")
    animaNyulStandardization = os.path.join(animaDir, "animaNyulStandardization")
    animaThrImage = os.path.join(animaDir, "animaThrImage")
    animaMaskImage = os.path.join(animaDir, "animaMaskImage")
    animaImageArithmetic = os.path.join(animaDir, "animaImageArithmetic")

    # Calls a command, if there are errors: outputs them and exit
    def call(command):
        command = [str(arg) for arg in command]
        status = subprocess.call(command)
        if status != 0:
            print(' '.join(command) + '\n')
            sys.exit('Command exited with status: ' + str(status))
        return status

    # Preprocess all patients: 
    #  - brain extraction
    #  - mask flair images with the union of the masks of both time points
    #  - bias correction
    #  - normalize (optional)

    print("Preprocessing patient " + vol_1 + "...")

    # Create the output directory which will contain the preprocessed files
    patientOutput = output

    masks = []

    flairs = ['flair_time01_on_middle_space.nii.gz', 'flair_time02_on_middle_space.nii.gz']
    groundTruths = ['ground_truth_expert1.nii.gz', 'ground_truth_expert2.nii.gz', 'ground_truth_expert3.nii.gz', 'ground_truth_expert4.nii.gz', 'ground_truth.nii.gz']

    # For both time points: extract brain
    for flairName in flairs:
        
        flair = os.path.join(patient, flairName)
        brain = os.path.join(patientOutput, flairName)
        mask = os.path.join(patientOutput, flairName.replace('.nii.gz', '_mask.nii.gz'))

        # Extract brain
        call(["python", animaBrainExtraction, "-i", flair, "--mask", mask, "--brain", brain, "-f", intermediateFolder])

        masks.append(mask)

    maskUnion = os.path.join(patientOutput, 'brain_mask.nii.gz')

    # Compute the union of the masks of both time points
    call([animaImageArithmetic, "-i", masks[0], "-a", masks[1], "-o", maskUnion])    # add the two masks
    call([animaThrImage, "-i", maskUnion, "-t", "0.5", "-o", maskUnion])                  # threshold to get a binary mask

    # Remove intermediate masks
    for mask in masks:
        os.remove(mask)

    # For both time points: mask, remove bias and normalize if necessary
    for flairName in flairs:
        
        flair = os.path.join(patient, flairName)
        brain = os.path.join(patientOutput, flairName)

        # Mask original FLAIR images with the union mask
        call([animaMaskImage, "-i", flair, "-m", maskUnion, "-o", brain])

        # Remove bias
        call([animaN4BiasCorrection, "-i", brain, "-o", brain, "-B", "0.3"])
        
        if templateFlair:
            if os.path.exists(templateFlair):
                # Normalize intensities with the given template
                call([animaNyulStandardization, "-m", brain, "-r", templateFlair, "-o", brain])
            else:
                print('Template file ' + templateFlair + ' not found, skipping normalization.')

def eval_prediction(vol_1, vol_2, output):
    # Defaults
    mode = 'other'
    input_type = 'time1_2_attn'
    norm_type = 'gaussian'
    img_type = 'patch'
    target_size = None
    img_shape = (64, 64)
    num_channels = 4
    interp_type = 'bilinear'

    # Evaluate models
    test_times = []
    # Load models
    model_1, _ = select_model({'img_shape': (*img_shape, num_channels)}, choice='scunet')
    model_2, _ = select_model({'img_shape': (*img_shape, num_channels)}, choice='scunet')
    model_3, _ = select_model({'img_shape': (*img_shape, num_channels)}, choice='scunet')
    model_4, _ = select_model({'img_shape': (*img_shape, num_channels)}, choice='scunet')
    model_5, _ = select_model({'img_shape': (*img_shape, num_channels)}, choice='scunet')

    # Get model names
    model_1.load_weights('/scunet/models/model_scunet_GDL_model1.hdf5')
    model_2.load_weights('/scunet/models/model_scunet_GDL_model2.hdf5')
    model_3.load_weights('/scunet/models/model_scunet_GDL_model3.hdf5')
    model_4.load_weights('/scunet/models/model_scunet_GDL_model4.hdf5')
    model_5.load_weights('/scunet/models/model_scunet_GDL_model5.hdf5')

    # Evaluate test volume
    elapsed_time = 0
    vol_1 = nib.load(vol_1)
    vol_2 = nib.load(vol_2)

    # Copy header information
    vol_2_header = vol_2.header.copy()
    vol_1_header = vol_1.header.copy()

    vol_2_affine = vol_2.affine.copy()

    vol_2 = vol_2.get_fdata()
    vol_1 = vol_1.get_fdata()

    start_time = time.time()

    # 3) Normalization
    if (norm_type == 'v2'):
        vol_2, _ = normalize_vol(vol_2, None, norm_type)
        vol_2 /= 1000  # normalize to small values

        vol_1, _ = normalize_vol(vol_1, None, norm_type)
        vol_1 /= 1000  # normalize to small values
    elif (norm_type == 'gaussian'):
        vol_1, _ = normalize_vol(vol_1, None, norm_type)
        vol_1_thresh = vol_1 > 0
        vol_1 = np.multiply(vol_1, vol_1_thresh)

        vol_2, _ = normalize_vol(vol_2, None, norm_type)
        vol_2_thresh = vol_1 > 0
        vol_2 = np.multiply(vol_2, vol_2_thresh)
    else:
        vol_1, _ = normalize_vol(vol_1, None, norm_type)

        vol_2, _ = normalize_vol(vol_2, None, norm_type)
    # Perform Difference Map Time2 - Time1 > 0
    diffMap = vol_2 - vol_1
    diffMask = diffMap > 0

    diffMap = np.multiply(diffMap, diffMask)

    if input_type == 'diff':
        diffMap, _ = normalize_vol(diffMap, None, 'min-max')
        input_vol = diffMap
    elif input_type == 'attn':
        vol_1_attn = np.multiply(vol_1, diffMap)
        vol_2_attn = np.multiply(vol_2, diffMap)
        attn_map = vol_1_attn + vol_2_attn
        attn_map, _ = normalize_vol(attn_map, None, 'min-max')
        input_vol = attn_map
    elif input_type == 'time1_attn':
        vol_1_attn = np.multiply(vol_1, diffMap)
        vol_1_attn, _ = normalize_vol(vol_1_attn, None, 'min-max')
        input_vol = vol_1_attn
    elif input_type == 'time2_attn':
        vol_2_attn = np.multiply(vol_2, diffMap)
        vol_2_attn, _ = normalize_vol(vol_2_attn, None, 'min-max')
        input_vol = vol_2_attn
        vol_1 = normalize_vol(vol_1, None, 'min-max')
        vol_1 = vol_1[0]
        vol_2 = normalize_vol(vol_2, None, 'min-max')
        vol_2 = vol_2[0]
    elif input_type == 'time1_2_attn':
        vol_1_attn = np.multiply(vol_1, diffMap)
        diffMap_1 = vol_1_attn
        vol_2_attn = np.multiply(vol_2, diffMap)
        diffMap_2 = vol_2_attn

    # Make predictions
    vol_size = (diffMap.shape[0:2])  # original dimensions

    # 2D Image predictions
    #4 channel
    pred_1, pred_2, pred_3, pred_4, pred_5 = get_pred_4ch_agg_patches(vol_1, vol_2, vol_1_attn, vol_2_attn, model_1, model_2, model_3,
                                                                      model_4, model_5, img_size=vol_size, plane='sag')

    pred_5agg = (pred_1 + pred_2 + pred_3 + pred_4 + pred_5) / 5

    pred_5agg = binarize(pred_5agg, 0.5)  # binarize pixels to 0 or 1
    # volshow(test_vol, pred)
    elapsed_time += (time.time() - start_time)

    # Save predictions
    test_pred_5agg = nib.Nifti1Image(pred_5agg, vol_2_affine, vol_2_header)
    nib.save(test_pred_5agg, output)

    print('\nPrediction saved!')

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Command line tool for new WML predictions.', add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='The time1, time2, and output filenames are required.')
    parser.add_argument('-t1', type=str, dest='vol_1', required=True, help="REQUIRED: Time1 path")
    parser.add_argument('-t2', type=str, dest='vol_2', required=True, help="REQUIRED: Time2 path")
    parser.add_argument('-o', type=str, dest='output', required=True, help="REQUIRED: Output path")
    parser.add_argument('-f', '--intermediate_folder', type=str, help="""Path where intermediate files (transformations, transformed images and rough mask) are stored 
                    (default is an temporary directory created automatically and deleted after the process is finished ;
                    intermediate files are deleted by default and kept if this option is given).
                    """)

    args = parser.parse_args()
    vol_1 = args.vol_1
    vol_2 = args.vol_2
    output = args.output
    if not os.path.exists(os.path.dirname(os.path.dirname(vol_1))):
        preprocess(vol_1)
    eval_prediction(vol_1, vol_2, output)