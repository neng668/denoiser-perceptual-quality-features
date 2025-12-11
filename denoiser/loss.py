# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Original copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Loss modules."""

import torch
import torch.nn.functional as F
import math
import torchaudio.transforms as T
import lpctorch as L


def complex_cepstrum(x, n=None, nanFlag=0):
    r"""Compute the complex cepstrum of a real sequence.

    Parameters
    ----------
    x : ndarray
        Real sequence to compute complex cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.

    Returns
    -------
    ceps : ndarray
        The complex cepstrum of the real data sequence `x` computed using the
        Fourier transform.
    ndelay : int
        The amount of samples of circular delay added to `x`.

    The complex cepstrum is given by

    .. math:: c[n] = F^{-1}\\left{\\log_{10}{\\left(F{x[n]}\\right)}\\right}

    where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.

    See Also
    --------
    real_cepstrum: Compute the real cepstrum.
    inverse_complex_cepstrum: Compute the inverse complex cepstrum of a real sequence.

    References
    ----------
    .. [1] Wikipedia, "Cepstrum".
           http://en.wikipedia.org/wiki/Cepstrum
    .. [2] M.P. Norton and D.G. Karczub, D.G.,
           "Fundamentals of Noise and Vibration Analysis for Engineers", 2003.
    .. [3] B. P. Bogert, M. J. R. Healy, and J. W. Tukey:
           "The Quefrency Analysis of Time Series for Echoes: Cepstrum, Pseudo
           Autocovariance, Cross-Cepstrum and Saphe Cracking".
           Proceedings of the Symposium on Time Series Analysis
           Chapter 15, 209-243. New York: Wiley, 1963.

    """

    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = unwrap2(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        #ndelay = torch.tensor(torch.round(unwrapped[..., center] / torch.tensor(math.pi)))
        ndelay = (torch.round(unwrapped[..., center] / torch.tensor(math.pi))).clone().detach()
        unwrapped -= torch.tensor(math.pi).cuda() * ndelay[..., None] * torch.arange(samples).cuda() / center
        return unwrapped, ndelay

    spectrum = torch.fft.fft(x, n=n)
    if nanFlag == 1:
        [loc1, loc2] = torch.where(spectrum == 0 + 0j)
        for errorCounter in range(list(loc1.size())[0]):
            spectrum[loc1[errorCounter].item(), loc2[errorCounter].item()] = 0.001
        #if list(loc1.size()) == [1]:


    unwrapped_phase, ndelay = _unwrap(torch.angle(spectrum))
    log_spectrum = torch.log(torch.abs(spectrum)) + 1j * unwrapped_phase
    ceps = torch.fft.ifft(log_spectrum).real

    return ceps


def diff(x, axis):
    """Take the finite difference of a tensor along an axis.
    Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.
    Returns:
    d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
    ValueError: Axis out of range for tensor.
    """
    shape = x.shape

    begin_back = [0 for unused_s in range(len(shape))]
#     print("begin_back",begin_back)
    begin_front = [0 for unused_s in range(len(shape))]

    begin_front[axis] = 1
#     print("begin_front",begin_front)

    size = list(shape)
    size[axis] -= 1
#     print("size",size)
    slice_front = x[begin_front[0]:begin_front[0]+size[0], begin_front[1]:begin_front[1]+size[1]]
    slice_back = x[begin_back[0]:begin_back[0]+size[0], begin_back[1]:begin_back[1]+size[1]]

#     slice_front = tf.slice(x, begin_front, size)
#     slice_back = tf.slice(x, begin_back, size)
#     print("slice_front",slice_front)
#     print(slice_front.shape)
#     print("slice_back",slice_back)

    d = slice_front - slice_back
    return d


def unwrap(p, discont=torch.tensor(math.pi), axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
    p: Phase tensor.
    discont: Float, size of the cyclic discontinuity.
    axis: Axis of which to unwrap.
    Returns:
    unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
    #     print("dd",dd)
    ddmod = torch.fmod(dd + torch.tensor(math.pi), 2.0 * torch.tensor(math.pi).cuda()) - torch.tensor(math.pi).cuda()  # ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
    #     print("ddmod",ddmod)

    idx = torch.logical_and(torch.eq(ddmod, -torch.tensor(math.pi).cuda()),
                         torch.gt(dd, 0))  # idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
    #     print("idx",idx)
    ddmod = torch.where(idx, torch.ones_like(ddmod) * torch.tensor(math.pi),
                     ddmod)  # ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
    #     print("ddmod",ddmod)
    ph_correct = ddmod - dd
    #     print("ph_corrct",ph_correct)

    idx = torch.lt(torch.abs(dd), discont)  # idx = tf.less(tf.abs(dd), discont)

    ddmod = torch.where(idx, torch.zeros_like(ddmod), dd)  # ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
    ph_cumsum = torch.cumsum(ph_correct, axis=axis)  # ph_cumsum = tf.cumsum(ph_correct, axis=axis)
    #     print("idx",idx)
    #     print("ddmod",ddmod)
    #     print("ph_cumsum",ph_cumsum)

    shape = torch.tensor(p.shape)  # shape = p.get_shape().as_list()

    shape[axis] = 1
    ph_cumsum = torch.cat([torch.zeros(shape.tolist(), dtype=p.dtype).cuda(), ph_cumsum], axis=axis)
    # ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
    #     print("unwrapped",unwrapped)
    return unwrapped


def unwrap2(phi, dim=-1):
    dphi = diff(phi, same_size=True)
    dphi_m = ((dphi+math.pi) % (2 * math.pi)) - math.pi
    dphi_m[(dphi_m==-math.pi)&(dphi>0)] = math.pi
    phi_adj = dphi_m-dphi
    phi_adj[dphi.abs()<math.pi] = 0
    return phi + phi_adj.cumsum(dim)


def diff(x, dim=-1, same_size=False):
    if same_size:
        return F.pad(x[..., 1:]-x[..., :-1], (1, 0))
    else:
        return x[..., 1:]-x[..., :-1]


def cepstrumStatistics(x, y, batch_size):
    """
    Calculates cepstrum statistics from input waveform tensors
    y: tensor containing sources signals
    x: tensor containing enhanced signals
    windowLength: default 80
    overlapSize: default 0.5
    TODO add kurtosis
    """
    n_start = 1
    x_std = torch.ones(batch_size, x.size(0)).cuda()
    y_std = torch.ones(batch_size, x.size(0)).cuda()
    x_skw = torch.ones(batch_size, x.size(0)).cuda()
    y_skw = torch.ones(batch_size, x.size(0)).cuda()
    x_krt = torch.ones(batch_size, x.size(0)).cuda()
    y_krt = torch.ones(batch_size, x.size(0)).cuda()
    cepStd = torch.ones(batch_size, x.size(0)).cuda()
    cepSkw = torch.ones(batch_size, x.size(0)).cuda()
    cepKrt = torch.ones(batch_size, x.size(0)).cuda()
    for k in range(batch_size):
        x_cep = complex_cepstrum(x[:, k, :], nanFlag=1)
        y_cep = complex_cepstrum(y[:, k, :])
        cep_diff = torch.nan_to_num(y_cep, neginf=-7.0, posinf=7.0) - x_cep
        cepStd[k, :] = torch.std(cep_diff, 1)
        cepSkw[k, :] = skew(cep_diff, 1)
        cepKrt[k, :] = kurtosis(cep_diff, 1)

    #return y_std, x_std, y_skw, x_skw, x_krt, y_krt
    return torch.mean(cepStd), torch.mean(cepSkw), torch.mean(cepKrt)


def cepstrumStatisticsSingle(x, y):
    """
    Calculates cepstrum statistics from input waveform tensors
    y: tensor containing sources signals
    x: tensor containing enhanced signals
    windowLength: default 80
    overlapSize: default 0.5
    """

    x_cep = complex_cepstrum(x, nanFlag=1)
    y_cep = complex_cepstrum(y, nanFlag=1)
    cep_diff = y_cep - x_cep
    cepStd = torch.std(cep_diff, 1)
    cepSkw = skew(cep_diff, 1)
    cepKrt = kurtosis(cep_diff, 1)

    return torch.mean(cepStd), torch.mean(cepSkw), torch.mean(cepKrt)


def LPCStatistics(x, y, lpc_prep):
    """
    Calculates LPC statistics from input waveform tensors
    y: tensor containing sources signals
    x: tensor containing enhanced signals
    windowLength: default 80
    overlapSize: default 0.5
    """

    x_lpc = lpc_prep(x)
    y_lpc = lpc_prep(y)
    lpc_diff = y_lpc - x_lpc
    lpc_diff_trimmed = lpc_diff[:, :, 1:lpc_diff.size()[2]]
    lpcStd = torch.std(lpc_diff_trimmed, 1)
    lpcSkw = skew(lpc_diff_trimmed, 1)
    lpcKrt = kurtosis(lpc_diff_trimmed, 1)

    return torch.mean(lpcStd), torch.mean(torch.abs(lpcSkw)), torch.mean(lpcKrt)


def calcEnergyDips(x, y, y_full, batch_size, windowLength, windowType, U, Fs):

    low_passband_hz = 150;
    high_passband_hz = 4500;
    low_passband = int(math.floor((low_passband_hz/(Fs/2)) * windowLength))
    high_passband = int(math.floor((high_passband_hz/(Fs/2)) * windowLength))

    MIN_SNR = -60
    MAX_SNR = 60

    # Set up VAD - 1920 for 16000 hz
    #first_120ms = y_full[:, 0:1920]
    #nsubframes = torch.floor(torch.tensor(1920) / (windowLength / 2)) - 1
    p = torch.zeros(y.size(0), 1)
    energyDips = torch.zeros(batch_size)
    energyExcess = torch.zeros(batch_size)

    for b in range(batch_size):
        #n_start_vad = 0
        #noise_ps = torch.zeros(windowLength, 1).cuda()
        #for j in range(int(nsubframes.item())):
        #    noise = first_120ms[b, n_start_vad: n_start_vad + windowLength]
        #    noise = noise * windowType
        #    noise_fft = torch.fft.fft(noise, windowLength)
        #    noise_ps = torch.squeeze(noise_ps) + abs(noise_fft ** 2/(windowLength*U))
        #    n_start_vad = n_start_vad + int(windowLength/2)
        #noise_ps = noise_ps/nsubframes.item()

        clean_speech = y[:, b, :]
        clean_speech = clean_speech*windowType
        clean_fft = torch.fft.fft(clean_speech, windowLength)
        clean_psd = (abs(clean_fft ** 2)/(windowLength * U))
        enhanced_speech = x[:, b, :]
        enhanced_speech = enhanced_speech*windowType
        enhanced_fft = torch.fft.fft(enhanced_speech, windowLength)
        enhanced_psd = (abs(enhanced_fft ** 2)/(windowLength * U))
        # simple VAD implementation
        p = torch.mean(clean_psd, 1)

        clean_energy_pb = torch.clamp(10*torch.log10(torch.sum(clean_psd[:, low_passband-1:high_passband-1], 1)), min=MIN_SNR, max=MAX_SNR)
        enhanced_energy_pb = torch.clamp(10*torch.log10(torch.sum(enhanced_psd[:, low_passband-1:high_passband-1], 1)), min=MIN_SNR, max=MAX_SNR)

        signal_energy = torch.sum(clean_psd, 1)
        noise_energy = torch.sum(enhanced_psd, 1)
        segmental_snr = 10*torch.log10(signal_energy/(noise_energy + 2.2204e-16) + 2.2204e-16)
        threshold = torch.clamp(clean_energy_pb - torch.exp(segmental_snr - 30)*0.2 - 10, min=MIN_SNR)

        firstVAD = torch.min((torch.where(torch.squeeze(p) > 0.0002))[0]).item()
        lastVAD = torch.max((torch.where(torch.squeeze(p) > 0.0002))[0]).item()

        #Median filter - TODO
        #energyDipAll = threshold - enhanced_energy_pb
        #energyExcessAll = enhanced_energy_pb - clean_energy_pb

        dipsArray = torch.where(enhanced_energy_pb[firstVAD:lastVAD] < threshold[firstVAD:lastVAD])[0] + firstVAD
        excessArray = torch.where(enhanced_energy_pb[firstVAD:lastVAD] > clean_energy_pb[firstVAD:lastVAD])[0] + firstVAD
        for k in range(dipsArray.size()[0]):
            energyDips[b] = energyDips[b] + threshold[dipsArray[k]] - enhanced_energy_pb[dipsArray[k]]

        for l in range(excessArray.size()[0]):
            energyExcess[b] = energyExcess[b] + enhanced_energy_pb[excessArray[l]] - clean_energy_pb[excessArray[l]]

        test = 1
        #VAD calculations
        #for k in range(y.size(0)):
        #    if k == 0:
        #        posteri = clean_psd[k, :]/noise_ps
        #        posteri_prime = torch.clamp(posteri - 1, min=0)
        #        priori = 0.98 + (1 - 0.98) * posteri_prime
        #    else:
        #        posteri = clean_psd[k, :] / noise_ps
        #        posteri_prime = torch.clamp(posteri - 1, min=0)
        #        priori = 0.98*(G ** 2) * posteri_prev + (1 - 0.98) * posteri_prime
        #    log_sigma_k = (posteri*priori)/(1 + priori) - torch.log10(1 + priori);
        #    p[k] = torch.clamp(torch.sum(log_sigma_k)/windowLength, max=1)
        #    G = (priori/(1 + priori))**0.5
        #    posteri_prev = posteri




        #energyDips = 1;
        #energyExcess = 1;

    return energyDips, energyExcess


def skew(a, axis=0, bias=True):
    """
    Calculates skewness in following matlab's scripts
    by default (bias=true), which is the same as matlab flag==1 which is default
    input a = tensor
    TODO - add bias corrections
    """
    x0 = a - torch.mean(a, axis).unsqueeze(1)
    s2 = torch.mean(torch.pow(x0, 2), axis)
    m3 = torch.mean(torch.pow(x0, 3), axis)
    s3 = m3 / torch.pow(s2, 1.5)
    return s3


def kurtosis(a, axis=0, bias=True):
    """
    Calculates skewness in following matlab's scripts
    by default (bias=true), which is the same as matlab flag==1 which is default
    input a = tensor
    TODO - add bias corrections
    """
    x0 = a - torch.mean(a, axis).unsqueeze(1)
    s2 = torch.mean(torch.pow(x0, 2), axis)
    m4 = torch.mean(torch.pow(x0, 4), axis)
    krt = m4 / torch.pow(s2, 2)
    return krt


class MFCCLoss(torch.nn.Module):
    """MFCC and MFCC std Loss module."""

    def __init__(self, sample_rate=16000, n_mfcc=5, n_fft=480, n_mels=5, hop_length=240, log_mels= True):
        """Initilize spectral convergence loss module."""
        super(MFCCLoss, self).__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.log_mels = True
        self.hop_length = hop_length
        self.mfcc_transform = T.MFCC(sample_rate, n_mfcc, log_mels= log_mels, melkwargs={"n_fft": n_fft, "n_mels": n_mels, "hop_length": hop_length,},).cuda()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Enhanced Speech Tensor
            y (Tensor): Clean Speech Tensor
        Returns:
            Tensor: MFCC std or MFCC .
        """
        x_mfcc = self.mfcc_transform(x)
        y_mfcc = self.mfcc_transform(y)
        MFCC_diff = x_mfcc - y_mfcc
        #return torch.mean(torch.abs(torch.std(y_mfcc, 2) - torch.std(x_mfcc, 2))), torch.mean(torch.abs(MFCC_diff))
        return torch.mean(torch.std(MFCC_diff, 2)), torch.mean(torch.abs(MFCC_diff))


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        #return F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        return torch.mean(torch.std((y_mag - x_mag), 1)), F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_std, mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss, mag_std


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.MFCC_losses = MFCCLoss(16000, 20, 480, 20, 240)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        mag_s_loss = 0.0
        mfcc_std = 0.0
        mfcc_avg = 0.0
        skw_loss = 0.0
        std_loss = 0.0
        krt_loss = 0.0
        dip_loss = 0.0
        excess_loss = 0.0

        # calculate phase of fft for x and y tensors - 160 for cep
        cep_window = 160
        cep_overlap = 0.50

        #below is very slow, try to reshape tensor to contain all the parts
        #n_start = 0
        #len1 = cep_window * (1 - cep_overlap)
        #nframes = math.floor(y.size(dim=1) / len1) - 1
        #x_windowed = torch.ones(nframes - 1, y.size(0), cep_window).cuda()
        #y_windowed = torch.ones(nframes - 1, y.size(0), cep_window).cuda()
        #for k in range(nframes - 1):
        #    x_windowed[k, 0:y.size(0), 0:0 + cep_window] = x[:, n_start:n_start+cep_window]
        #    y_windowed[k, 0:y.size(0), 0:0 + cep_window] = y[:, n_start:n_start + cep_window]
        #    n_start = int(n_start + cep_window*(1-cep_overlap))

        #hammingWin = torch.hamming_window(cep_window).cuda()
        #U = sum(hammingWin**2)/cep_window

        #sr = 16000
        #frame_duration = 0.01
        #frame_overlap = 0.5
        #K = 8
        #lpc_prep = L.LPCCoefficients(sr, frame_duration, frame_overlap, order=(K - 1))
        #lpc_std, lpc_skw, lpc_krt = LPCStatistics(x, y, lpc_prep)

        #for b in range(y.size(0)):
        #    clean_speech = y_windowed[:, b, :]
        #    clean_speech = clean_speech*hammingWin
        #    clean_fft = torch.fft.fft(clean_speech, cep_window)
        #    clean_psd = (abs(clean_fft ** 2)/(cep_window * U))
        #    p = torch.mean(clean_psd, 1)
        #    firstVAD = torch.min((torch.where(torch.squeeze(p) > 0.0002))[0]).item()
        #    lastVAD = torch.max((torch.where(torch.squeeze(p) > 0.0002))[0]).item()
        #    complex_cepstrum(torch.unsqueeze(y_windowed[0, b, :], 0))
        #    [std_l, skw_l, krt_l] = cepstrumStatisticsSingle(x_windowed[firstVAD:lastVAD, b, :], y_windowed[firstVAD:lastVAD, b, :])
        #    std_loss += std_l
        #    skw_loss += skw_l
        #    krt_loss += krt_l

        #std_loss /= y.size(0)
        #skw_loss /= y.size(0)
        #krt_loss /= y.size(0)

        #[energyDips, energyExcess] = calcEnergyDips(x_windowed, y_windowed, y, y.size(0), cep_window, hammingWin, U, 16000)
        #energyDips = torch.log10(torch.mean(energyDips) + 1)
        #energyExcess = torch.log10(torch.mean(energyExcess) + 1)
        #[y_std, x_std, y_skw, x_skw, x_krt, y_krt] = cepstrumStatistics(x_windowed, y_windowed, y.size(0))
        #[std_loss, skw_loss, krt_loss] = cepstrumStatistics(x_windowed, y_windowed, y.size(0))
        #std_loss = F.l1_loss(y_std, x_std)
        #skw_loss = F.l1_loss(y_skw, x_skw)
        #krt_loss = F.l1_loss(y_krt, x_krt)
        std_loss = 1
        skw_loss = 1
        krt_loss = 1

        #for b in range(y.size(0)):
        #    clean_speech = y_windowed[:, b, :]
        #    clean_speech = clean_speech*hammingWin
        #    clean_fft = torch.fft.fft(clean_speech, cep_window)
        #    clean_psd = (abs(clean_fft ** 2)/(cep_window * U))
        #    p = torch.mean(clean_psd, 1)
        #    firstVAD = torch.min((torch.where(torch.squeeze(p) > 0.0002))[0]).item()
        #    lastVAD = torch.max((torch.where(torch.squeeze(p) > 0.0002))[0]).item()
        #    #convert framenumber to index
        #    firstVADIndex = int((cep_window/2)*firstVAD)
        #    lastVADIndex = int((cep_window/2)*lastVAD)
        #    mfcc_std_one, mfcc_avg_one = self.MFCC_losses(torch.unsqueeze(x[b, firstVADIndex:lastVADIndex],0), torch.unsqueeze(y[b, firstVADIndex:lastVADIndex],0))
        #    mfcc_std += mfcc_std_one
        #    mfcc_avg += mfcc_avg_one
        #mfcc_std /= y.size(0)
        #mfcc_avg /= y.size(0)

        #mfcc_std, mfcc_avg = self.MFCC_losses(x, y)

        #for f in self.stft_losses:
        #    sc_l, mag_l, mag_s = f(x, y)
        #    sc_loss += sc_l
        #    mag_loss += mag_l
        #    mag_s_loss += mag_s
        #sc_loss /= len(self.stft_losses)
        #mag_loss /= len(self.stft_losses)
        #mag_s_loss /= len(self.stft_losses)
        sc_loss = 1
        mag_loss = 1
        mag_s_loss = 1
        lpc_std = 1

        #return self.factor_sc * sc_loss, self.factor_mag * mag_loss, energyDips/100, energyExcess/100
        return self.factor_sc*sc_loss, self.factor_mag*mag_loss, self.factor_mag*mag_s_loss, self.factor_mag*mfcc_std, self.factor_mag*mfcc_avg, std_loss*0.01, skw_loss*self.factor_mag, krt_loss*self.factor_mag, lpc_std*0.1
        #return self.factor_sc * sc_loss, self.factor_mag * mag_loss, self.factor_mag * mag_s_loss, std_loss * 0.025, skw_loss * 0.0025, krt_loss * 0.000125

