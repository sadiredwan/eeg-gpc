import os
import mne
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from tqdm import tqdm
import pickle
mne.set_log_level(40)


epochs_list, y = [], []
df = pd.read_csv('data/participants.tsv', sep='\t')
label_map = {'Control': 0, 'Psychosis': 1}
sublist = df['participant_id']
typelist = df['type']
labels = dict((s, label_map[t]) for s, t in zip(sublist, typelist))
for f in tqdm(os.listdir('data/processed/epochs')):
    epochs = mne.read_epochs('data/processed/epochs/'+f)
    epochs.drop_channels(epochs.info['ch_names'][60:])
    resampled = epochs.get_data()
    resampled = mne.filter.resample(resampled, down=3)
    epochs = mne.EpochsArray(resampled, info=epochs.info)
    epochs_list.append(epochs)
    for epoch in epochs.get_data():
        y.append(labels[f.replace('.fif', '')])
y = np.array(y)

def eeg_power_band(epochs):
    FREQ_BANDS = {
        'delta': [0.5, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'sigma': [12, 16]
    }
    spectrum = epochs.compute_psd(picks='eeg', fmin=0.5, fmax=30.)
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds /= np.sum(psds, axis=-1, keepdims=True)
    X = []
    for fmin, fmax in tqdm(FREQ_BANDS.values()):
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    return np.concatenate(X, axis=1)

X = eeg_power_band(mne.concatenate_epochs(epochs_list))
X, y = BorderlineSMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

gpc_pipe = make_pipeline(
    GaussianProcessClassifier(1.0 * RBF(1.0))
)
gpc_pipe.fit(X_train, y_train)
y_pred = gpc_pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy score (GPC): {}'.format(acc))

np.save('data/test/X_test.npy', X_test)
np.save('data/test/y_test.npy', y_test)

filename = 'models/gpc_pipe.sav'
pickle.dump(gpc_pipe, open(filename, 'wb'))
