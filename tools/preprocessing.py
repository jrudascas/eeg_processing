import mne
import numpy as np
from mne.preprocessing import ICA

def change_channels_labels(data):
    # Electrodes are named with LETTER-NUMBER combinations (A1, B2, F4, …) (65+3 locations)
    data = data.copy()
    new_labels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7',
                  'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz',
                  'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2',
                  'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8',
                  'PO4', 'O2']

    for label_id in range(len(new_labels)):
        data.info['ch_names'][label_id] = new_labels[label_id]
        data.info['chs'][label_id]['ch_name'] = new_labels[label_id]

    montage = mne.channels.read_montage('biosemi64')
    data.info['bads'] = list(set(data.info['ch_names']) - set(montage.ch_names))

    data.set_montage(montage)
    data.plot_sensors(show_names=True)

    # raw_data.info['bads']

    # Set EOG chanels
    data.set_channel_types(mapping={'Fp1': 'eog'})
    return data

def apply_reference(data):
    data = data.copy()
    data.set_eeg_reference('average', projection=True)
    data.apply_proj()
    return data


def filter(data, fmin, fmax):
    data = data.copy()
    # band-pass filtering in the range 1 Hz - 5 Hz
    #fmin, fmax = 1., 40.
    data.filter(fmin, fmax, n_jobs=1, fir_design='firwin')
    # raw_data.plot(n_channels=5, title='Raw data')
    return data

def reade_epochs(data, tmin, tmax, evend_id_path):
    data = data.copy()
    #tmin = -0.1  # start of each epoch (100ms before the trigger)
    #tmax = 0.2  # end of each epoch (200ms after the trigger)
    #event_id = {'auditory/left': 10, 'auditory/right': 11, 'visual/left': 12, 'rest': 13}
    events = mne.find_events(data)
    evend_id_data = np.loadtxt(evend_id_path, delimiter=';')
    events[:, 2] = evend_id_data
    event_id = np.unique(evend_id_data)
    baseline = (None, 0)  # means from the first instant to t = 0

    picks = mne.pick_types(data.info, meg=False, eeg=True, eog=True, stim=False, exclude='bads')

    # Read epochs
    epochs = mne.Epochs(data, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=baseline, reject_by_annotation=True)

    print('Removing bad epochs')
    epochs.drop_bad()
    return epochs.copy()

def remove_eof_local_detection(data):
    data = data.copy()
    event_id = 998
    eog_events = mne.preprocessing.find_eog_events(data, event_id)
    n_blinks = len(eog_events)

    # Read epochs
    picks = mne.pick_types(data.info, meg=False, eeg=False, stim=False, eog=True, exclude='bads')
    tmin, tmax = -0.2, 0.2
    blink_epochs = mne.Epochs(data, eog_events, event_id, tmin, tmax, picks=picks)
    data = blink_epochs.get_data()
    print("Number of detected EOG artifacts : %d" % len(blink_epochs))

    # Center to cover the whole blink with full duration of 0.5s:

    onset = eog_events[:, 0] / data.info['sfreq'] - 0.25
    duration = np.repeat(0.5, n_blinks)

    annot = mne.Annotations(onset, duration, ['bad blink'] * n_blinks, orig_time=data.info['meas_date'])
    data.set_annotations(annot)
    data.plot(n_channels=5, events=eog_events)
    return data


def remove_eof_ica(data):
    data = data.copy()
    data.filter(1., None, n_jobs=1, fir_design='firwin')

    picks_eeg = mne.pick_types(data.info, meg=False, eeg=True, eog=True, stim=False, exclude='bads')
    ica = ICA(n_components=20, method='fastica', random_state=23)
    ica.fit(data, picks=picks_eeg, decim=3)

    ica.plot_components(title='ICA components BEFORE to remove blink artifacts')

    # Cheking the properties of first three independent components
    # ica.plot_properties(raw_data, picks=[0, 1, 2], psd_args={'fmax': 35.})

    ica.exclude.extend([0, 1])
    ica.apply(data)

    # Run again ICA to check the results
    data.filter(1., None, n_jobs=1, fir_design='firwin')

    picks_eeg = mne.pick_types(data.info, meg=False, eeg=True, eog=True, stim=False, exclude='bads')
    ica = ICA(n_components=20, method='fastica', random_state=23)
    ica.fit(data, picks=picks_eeg, decim=3)

    ica.plot_components(title='ICA components AFTER to remove blink artifacts')
    return data


def remove_eof_artifacts(data, type='ica'):
    if type == 'ica':
        resutl = remove_eof_ica(data)
    elif type == 'local':
        resutl = remove_eof_local_detection(data)
    return resutl