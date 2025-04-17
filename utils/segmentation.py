import numpy as np
import librosa
from sklearn.mixture import GaussianMixture


class Segmentation:
    def __init__(self, sr=16000, frame_length=0.025, hop_length=0.010, n_mfcc=13, gmm_components=3):
        self.sr = sr
        self.frame_length = int(frame_length * sr)
        self.hop_length = int(hop_length * sr)
        self.n_mfcc = n_mfcc
        self.gmm_components = gmm_components

    def extract_features(self, audio):
        """
        Extract MFCC features from audio.
        """
        mfcc = librosa.feature.mfcc(y=audio,
                                    sr=self.sr,
                                    n_mfcc=self.n_mfcc,
                                    n_fft=self.frame_length,
                                    hop_length=self.hop_length)
        return mfcc.T  # Shape: (frames, n_mfcc)

    def bic_score(self, X1, X2):
        """
        Compute the Bayesian Information Criterion (BIC) between two segments.
        Lower BIC indicates a better fit.
        Equation (6): BIC = -2 * logL + k * log(N)
        """
        X_combined = np.concatenate([X1, X2], axis=0)
        N = X_combined.shape[0]

        # GMM for combined segment
        gmm_combined = GaussianMixture(n_components=self.gmm_components, covariance_type='full')
        gmm_combined.fit(X_combined)
        logL_combined = gmm_combined.score(X_combined) * N
        k_combined = self._num_params(gmm_combined)

        # GMM for each segment
        gmm_1 = GaussianMixture(n_components=self.gmm_components, covariance_type='full')
        gmm_2 = GaussianMixture(n_components=self.gmm_components, covariance_type='full')
        gmm_1.fit(X1)
        gmm_2.fit(X2)
        logL_sep = gmm_1.score(X1) * len(X1) + gmm_2.score(X2) * len(X2)
        k_sep = self._num_params(gmm_1) + self._num_params(gmm_2)

        bic_combined = -2 * logL_combined + k_combined * np.log(N)
        bic_separated = -2 * logL_sep + k_sep * np.log(N)

        return bic_combined - bic_separated

    def _num_params(self, gmm):
        """
        Count number of parameters in a GMM.
        """
        cov_params = gmm.n_components * gmm.means_.shape[1] * gmm.means_.shape[1]
        mean_params = gmm.means_.size
        weight_params = gmm.weights_.size
        return cov_params + mean_params + weight_params

    def segment_audio(self, audio, window_size=3.0, step_size=1.0, bic_threshold=0.0):
        """
        Perform segmentation using sliding window and BIC score.
        """
        features = self.extract_features(audio)
        segment_boundaries = [0]

        window_frames = int(window_size / (self.hop_length / self.sr))
        step_frames = int(step_size / (self.hop_length / self.sr))

        for start in range(0, len(features) - 2 * window_frames, step_frames):
            end1 = start + window_frames
            end2 = end1 + window_frames
            X1 = features[start:end1]
            X2 = features[end1:end2]

            bic = self.bic_score(X1, X2)

            if bic > bic_threshold:  # Higher BIC -> more likely two different speakers
                segment_boundaries.append(end1)

        segment_boundaries.append(len(features))
        return segment_boundaries
