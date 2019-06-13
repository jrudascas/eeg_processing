#pip install mne numpy nibabel matplotlib nilearn pyqt5 mayavi

import mne
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from mne.preprocessing import ICA
from mne.viz import plot_arrowmap
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.viz import plot_evoked_topo
from mne.minimum_norm import make_inverse_operator, apply_inverse

from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs
from mne.connectivity import spectral_connectivity, seed_target_indices
#from IPython.display import HTML
from mne.datasets import sample
from mne.stats import permutation_cluster_test
from mne.stats import permutation_t_test
from mne.time_frequency import AverageTFR
import os
from natsort import natsorted


subjects_path = ''

for dirs in natsorted(os.listdir(subjects_path)):
    print('Processing: ' + dirs)

# Reading raw data
raw_data = mne.io.read_raw_edf(raw_data_file, preload=True)

