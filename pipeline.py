#pip install mne numpy nibabel matplotlib nilearn pyqt5 mayavi

import mne
import os
from natsort import natsorted
from .tools.preprocessing import change_channels_labels, apply_reference, filter, remove_eof_artifacts
from .tools.frequency_analysis import cross_spectral_density_analysis, power_spectral_density_analysis

subjects_path = '/'
filter_bounds = [1., 40.]

for dirs in natsorted(os.listdir(subjects_path)):
    print('Processing: ' + dirs)

    subject_path = subjects_path + '/' + dirs
    # Reading raw data
    raw_data = mne.io.read_raw_edf(subject_path, preload=True)

    #Preprocessing
    ##################################################################
    raw_data = change_channels_labels(raw_data)

    raw_data = apply_reference(raw_data)

    raw_data = filter(raw_data, filter_bounds[0], filter_bounds[1])

    raw_data = remove_eof_artifacts(raw_data, type='ica')

    #Frequency analisys
    #################################################################



