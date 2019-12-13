#-*-coding:utf-8 -*-
import zipfile
import os

def build_submission(filename, file_dir, file_list, optional_file_list=()):
    """Helper utility to check homework assignment submissions and package them.

    Parameters
    ----------
    filename : str
        Output zip file name
    file_dir : str
        dir of the necessary files.
    file_list : tuple
        Tuple of files to include
    """
    # check each file exists
    for part_i, file_i in enumerate(file_list):
        if not os.path.exists(file_i):
            print('\nYou are missing the file {}.  '.format(file_i) +
                  'It does not look like you have completed Part {}.'.format(
                part_i + 1))

    def zipdir(path, zf):
        for root, dirs, files in os.walk(path):
            for file in files:
                # make sure the files are part of the necessary file list
                if file.endswith(file_list) or file.endswith(optional_file_list):
                    zf.write(os.path.join(root, file))

    # create a zip file with the necessary files
    zipf = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)
    zipdir(file_dir, zipf)
    zipf.close()
    print('Your assignment zip file has been created!')
    print('Now submit the file:\n{}\nto Kadenze for grading!'.format(
        os.path.abspath(filename)))