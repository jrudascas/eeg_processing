import mne
from mne.time_frequency import csd_fourier, csd_multitaper


def power_spectral_density_analysis(data, plot_path = None):
    data = data.copy()
    fig = data.plot_psd(area_mode='range', tmax=10.0, show=False, average=True)
    if plot_path is not None:
        fig.savefig(plot_path, dpi=1200)

def cross_spectral_density_analysis(epochs, fmin, fmax):
    epochs = epochs.copy()
    #event_id = 998
    #eog_events = mne.preprocessing.find_eog_events(data, event_id)
    #n_blinks = len(eog_events)

    # Read epochs
    #picks = mne.pick_types(data.info, eeg=True, exclude='bads')
    #tmin, tmax = -0.2, 0.2
    #epochs_eog = mne.Epochs(data, eog_events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), preload=True)

    csd_fft = csd_fourier(epochs, fmin=fmin, fmax=fmax, n_jobs=2)
    #csd_mt = csd_multitaper(epochs_eog, fmin=15, fmax=20, adaptive=True, n_jobs=1)

    #frequencies = [16, 17, 18, 19, 20]
    #csd_wav = csd_morlet(epochs_eog, frequencies, decim=10, n_jobs=1)
    return csd_fft
    #csd_fft.mean().plot()
    #plt.suptitle('short-term Fourier transform')

    #csd_mt.mean().plot()
    #plt.suptitle('adaptive multitapers')

    #csd_wav.mean().plot()
    #plt.suptitle('Morlet wavelet transform')