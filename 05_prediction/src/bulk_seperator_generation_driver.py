import os
import sys
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm
from colorama import Fore, Style
import tensorflow as tf

from BaseObjects import Page
from Predictors.NetPredictor import NetPredictor
from Predictors.VotingPredictor import VotingPredictor
from utils import load

"""
Used to check that the version of TF is what we expect. Otherwise we warn the user
"""
TENSORFLOW_VERSION = '1.15'
"""
The filename of the file that contains the list of all saved outputs from the model
"""
OUTPUT_LIST_FILE_NAME = 'generated_file_list.csv'
"""
Name of file where errors are saved to
"""
OUTPUT_ERR_LIST_FILE_NAME = 'image_error_list.csv'


def bulk_generate_separators(image_dir: str, image_ext: str, output_dir: str, model_dir: str, regenerate: bool = False,
                             debug: bool = False, verbose: bool = False) -> int:
    """
    A simple wrapper function to call the model and run it against the provided inputs. Tries to fail gracefully
    as much as possible. Offers several degrees of output and debug info. Attempts to provide progress on the entire job too.
    Args:
        image_dir (): The directory where all images are stored. Not a recursive search
        image_ext (): The extension of images desired to be labeled. Supports single extension only
        output_dir (): The directory to save all outputs, file lists, labeled images etc
        model_dir (): The directory that contains the model to be loaded
        regenerate (): If set to true images will be re-labeled even if their output file already exists in the folder
        debug (): Prints some extra info and saved the colorized labeled images for debug purposes
        verbose (): Increases the amount of output from the method and underlying libraries(mostly TF)

    Returns:
        The number of images labeled.

    """
    if not Path(image_dir).exists():
        print(f'{Fore.RED} The the following image directory could not be found. Exiting...')
        print(f'Directory: {image_dir}')
        print(Style.RESET_ALL)
        sys.exit(1)
    if not Path(image_dir).exists():
        print(f'{Fore.RED} The the following output directory could not be found. Exiting...')
        print(f'Directory: {output_dir}')
        print(Style.RESET_ALL)
        sys.exit(1)
    if not Path(image_dir).exists():
        print(f'{Fore.RED} The the following model directory could not be found. Exiting...')
        print(f'Directory: {model_dir}')
        print(Style.RESET_ALL)
        sys.exit(1)
    if TENSORFLOW_VERSION not in tf.__version__:
        # Tensorflow version does not match, warn
        print(f'{Fore.LIGHTYELLOW_EX}Tensorflow version does not match expected: {TENSORFLOW_VERSION}')
        print('continue at your own risk, but things might not work correctly!')
        print(Style.RESET_ALL)
    if verbose:
        print('preparing to load model from file...')
        # This sets the logging level to all messages from TF
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        # Hides info level log messages from TF. Shows everything from WARNING upward.
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    try:
        # Load the model from files
        loaded = load([
            (NetPredictor, "v3/sep/1"),
            (NetPredictor, "v3/sep/2"),
            (NetPredictor, "v3/sep/4")], models_path=model_dir)

        # Init objects with the loaded models
        sep_predictor = VotingPredictor(
            loaded["v3/sep/1"],
            loaded["v3/sep/2"],
            loaded["v3/sep/4"])
    except BaseException as e:
        print(f'{Fore.RED}there was an error while loading the model from the provided directory{Fore.RESET}')
        raise e
    labeled_image_count = 0
    # Scan through provided input folder for all images
    file_list = list(Path(image_dir).glob(f'*.{image_ext}'))
    # Lists to track images that are labeled(or not)
    image_success_list: List[str] = []
    image_err_list: List[str] = []
    # Loop over all images and perform labeling actions
    for image in tqdm(file_list, 'labeling images..'):
        image_stem = Path(image).stem
        if not regenerate:
            if Path(output_dir).joinpath(image_stem + '.sep.npy').exists():
                # Output already generated, skip this one
                continue

        # Check if image exists first and that we can indeed open it.
        # This allows us to fail gracefully and catch image level issues and not have them
        # stop the entire operation.
        try:
            Image.open(image)
        except Exception:
            print(f'{Fore.RED} could not load image {str(image)}')
            print(Fore.RESET)
            image_err_list.append(str(image))
            continue
        page = Page.Page(image)
        model_result = sep_predictor(page)
        if debug:
            # In debug mode we save the actual labeled images
            model_result.save(Path(output_dir).joinpath(image_stem + '.sep.png'))
        # Save labels as a numpy array
        output_file = Path(output_dir).joinpath(image_stem)
        image_success_list.append(str(output_file))
        model_result.save_labels_array(output_file, page.shape)
        labeled_image_count += 1

    # Save the list of files we'eve generated during this round to a file
    output_file_list_path = Path(output_dir).joinpath(OUTPUT_LIST_FILE_NAME)
    output_file_err_list_path = Path(output_dir).joinpath(OUTPUT_ERR_LIST_FILE_NAME)
    with open(output_file_list_path, 'a+') as file:
        file.writelines(image_success_list)
    with open(output_file_err_list_path, 'w') as file:
        file.writelines(image_err_list)
    print(f'appended saved output file list to: {output_file_list_path} ')
    print(f'saved error output file list to: {output_file_err_list_path} ')
    return labeled_image_count
