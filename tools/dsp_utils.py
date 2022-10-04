from scipy import signal
from scipy.fft import fft, fftshift
import numpy as np
import matplotlib.pyplot as plt
import abc


class SigFilter(metaclass=abc.ABCMeta):
    """
    Abstract class for Low pass filter. Even if it is possible to use them as pass-band or high-pass filter,
    in this framework the filters are used to clear the signals from high frequency noise. So we use them as LPFs.

    Parameters:
        - delta_t_sig : delta t of the signal (time series) to be filtered [Sec].
        - ripple_bs : the desired attenuation in the stop band [dB].
        - cutoff_hz: cutoff frequency of the filter [Hz].
        - transition_width: the desired transition width between the pass band and the stop band. (In
        the ideal case it is = 0) [Hz].

    output:
        - x_filtered : the filtered version of the signal
    Notes:
       We use the delta_t of the signals instead of the sampling rate fs. This is just a designing choice
       : fs = 1 / dt .
       According to the Shannon theorem the sample rate fs >= 2B, where B is the bandwith of the signal.
       To obtain the nyquist rate of a time series sampled at frequency dt the nyquist rate = 1/(2*dt)

    """

    def __init__(self, fs: float, cutoff_hz: float, f_name: str = 'NoName'):
        self.nyquist_rate = fs / 2                          # fs/2
        self.cutoff_norm = cutoff_hz / self.nyquist_rate    # Normalized cut-off frequency
        self.a, self.b = self._initialize_filter()          # Compute the numerator (b) and denominator (a) coefficients
        self.filter_name = f_name                        # Name and ID of the filter: for verbose plots

    @abc.abstractmethod
    def _initialize_filter(self, **kwargs) -> (np.ndarray, np.ndarray):
        raise NotImplementedError

    def filter(self, x_unfiltered) -> np.ndarray:
        #  A typical use of the function  'lfilter_zi' is to set the initial state so that the
        #     output of the filter starts at the same value as the first element of
        #     the signal to be filtered.

        # Evaluate the padding length
        x_len = len(x_unfiltered)
        pad_len = 3 * max(len(self.a), len(self.b))
        if x_len < pad_len:
            safe_padlen = int(np.floor(0.5 * x_len))
            print('Warning during filtering, the length of the original signal ({}) '
                  'is lower than the suggested padding length: ({}), using the half of the length of the signal: {}'
                  .format(x_len, pad_len, safe_padlen))
            pad_len = safe_padlen

        x_filtered = signal.filtfilt(self.b, self.a, x_unfiltered, padlen=pad_len)
        return x_filtered

    def disp_info(self):
        # Set the font sizes for the plot
        title_size = 13
        axis_size = 12

        # Compute frequency response of the filter
        # using signal.freqz function
        wz, hz = signal.freqz(self.b, self.a)

        # Calculate Magnitude from hz in dB
        mag = 20 * np.log10(abs(hz))
        min_mag = np.min(mag) + 0.1 * np.min(mag)
        max_mag = np.max(mag) + 5

        # Calculate phase angle in degree from hz
        phase = np.unwrap(np.arctan2(np.imag(hz), np.real(hz))) * (180 / np.pi)

        # Calculate frequency in Hz from wz
        freq = (wz * (self.nyquist_rate * 2)) / (2 * np.pi)

        # Calculate the cut-off back from normalization
        cut_off_f = self.cutoff_norm * self.nyquist_rate

        # Plot filter magnitude and phase responses using subplot.
        fig = plt.figure(figsize=(10, 6))

        # Plot Magnitude response
        sub1 = plt.subplot(4, 1, 1)
        sub1.plot(freq, mag)
        sub1.axis([1, self.nyquist_rate, min_mag, max_mag])
        sub1.set_xlabel('Frequency [Hz]', fontsize=axis_size)
        sub1.set_ylabel('Magnitude [dB]', fontsize=axis_size)
        sub1.set_title(self.filter_name + ' filter frequency response', fontsize=title_size)
        sub1.margins(0, 0.1)
        sub1.grid(which='both', axis='both')
        sub1.axvline(cut_off_f, color='green')  # cutoff frequency


        # Plot phase angle
        sub2 = plt.subplot(4, 1, 2)
        sub2.plot(freq, phase, 'g', linewidth=2)
        sub2.set_ylabel('Phase (degree)', fontsize=axis_size)
        sub2.set_xlabel(r'Frequency (Hz)', fontsize=axis_size)
        sub2.set_title(self.filter_name + r' Phase response', fontsize=title_size)
        sub2.grid()

        #  calculate impulse response and step response of the filter
        # Define the impulse sequence of length 60
        impulse = np.repeat(0., 60)
        impulse[0] = 1.
        x = np.arange(0, 60)

        # Compute the impulse response
        response = self.filter(impulse)

        # Plot filter impulse and step response:
        sub3 = plt.subplot(4, 1, 3)
        sub3.stem(x, response, 'm', use_line_collection=True)
        sub3.set_ylabel('Amplitude', fontsize=axis_size)
        sub3.set_xlabel(r'n (samples)', fontsize=axis_size)
        sub3.set_title(self.filter_name + r' Impulse response', fontsize=title_size)
        sub3.grid()

        sub4 = plt.subplot(4, 1, 4)
        step = np.cumsum(response)  # Compute step response of the system
        sub4.stem(x, step, 'g', use_line_collection=True)
        sub4.set_ylabel('Amplitude', fontsize=axis_size)
        sub4.set_xlabel(r'n (samples)', fontsize=axis_size)
        sub4.set_title(self.filter_name + r' Step response', fontsize=title_size)
        sub4.grid()

        plt.subplots_adjust(hspace=0.5)
        fig.tight_layout()
        plt.show()


class FIRLowPass(SigFilter):
    """
    FIR (Finite impulse response) Low pass filter.
    In this case we use the kaiser filter to obtain the order and the parameters for the filter;
    keep in mind that other types of windowing algorithms exist: Hamming, Hanning, Rectangular, Blackman.
    We'll use scipy: lfilter, firwin and kaiserord

    Note:   The FIR filters of length M introduces a transient of length M that must be discarded at the beginning and
            at the end of the signal .
            - delay :   the phase delay of the filter, it used with the length of the filter to select the reliable
                        part of the filtered signal.
            Example of usage:
                    - time delay compensation: t[filter.N - 1:] - delay
                    - signal transitory removing: x_filtered[filter.N - 1:]
    """

    def __init__(self, fs: float, cutoff_hz: float, ripple_db: float, transition_width: float, id_filter: str = ''):
        self.ripple = ripple_db                       # Attenuation on the stop band [dB]
        self.transition_width = transition_width      # the width between the pass and stop band [Hz]
        filter_name = 'FIR_{} '.format(id_filter)      # name of the FIR filter
        super().__init__(fs, cutoff_hz, filter_name)

    def _initialize_filter(self):

        #  Compute the order and the Kaiser window parameter for the FIR filter
        transition_width_norm = self.transition_width / self.nyquist_rate
        self.M, self.beta = signal.kaiserord(self.ripple, transition_width_norm)

        #  Using the firwin method it is possible to obtain the impulse response of the filter using the selected
        #  windowing method.
        b = signal.firwin(self.M, self.cutoff_norm, window=('kaiser', self.beta))

        a = [1.0]  # In this case (FIR) the (a) denominator coefficient array will be only 1.0

        #  the filtering process will introduce a fixed M delay due to the phase shift
        self.delay = 0.5 * (self.M - 1) / (self.nyquist_rate * 2)

        return a, b

    #  Return the phase delay of the filter
    def get_delay(self):
        return self.delay

    # Return the length of the filter
    def get_filter_len(self):
        return self.M


class BtwLowPass(SigFilter):
    def __init__(self, fs: float, cutoff_hz: float, order: int,  id_filter: str = ''):
        filter_name = 'BTW_{} '.format(id_filter)            # name of the FIR filter
        self.order = order
        super().__init__(fs, cutoff_hz, filter_name)

    def _initialize_filter(self):
        #  Inizialize the Butterworth Filter as Digital IIR
        b, a = signal.butter(N=self.order, Wn=self.cutoff_norm, btype='low', analog=False)
        return a, b


class NotchIir(SigFilter):
    def __init__(self, fs: float, f_notch: float, q_fact: float, id_filter: str = ''):
        filter_name = 'Notch_{} '.format(id_filter)  # name of the Notch iir filter
        self.q = q_fact
        super().__init__(fs, f_notch, filter_name)

    def _initialize_filter(self):
        #  Inizialize the Notch Filter as Digital IIR
        b, a = signal.iirnotch(w0=self.cutoff_norm, Q=self.q)
        return a, b


class Tools:

    @staticmethod
    def get_kaiser_window(side_lobe_amplitude: float, fs: float, resolution: float, display=False):
        nyq_freq = fs/2                 # equivalent to pi (rad/samples)
        dml = (np.pi*resolution) / nyq_freq
        asl = side_lobe_amplitude
        # Evaluate beta
        if (asl > 0) and (asl <= 13.26):
            beta = 0
        elif (asl > 13.26) and (asl <= 60):
            beta = 0.76608*(asl - 13.26)**0.4 + 0.09834*(asl - 13.26)
        elif (asl > 60) and (asl <= 120):
            beta = 0.12438*(asl+6.3)
        else:
            raise ValueError('The side lobe amplitude must be between 0 and 120')
        # Evaluate the length of the window
        """
        iven relative side lobe amplitude ASL and main lobe width delta_ML
        specifications, we now need to compute the window length L to satisfy the
        main lobe width specification.
        An approximate relation between L, ASL, and delta_ML is provided by:
        L = 24*pi(ASL + 12) / (155*delta_ML)
        """
        L = int(24*np.pi*(asl+12)/(155*dml))

        # Create the Kaiser windows
        window = signal.windows.kaiser(L, beta=beta)

        if display:
            plt.figure(figsize=(10, 6))
            plt.subplot(211)
            plt.plot(window)
            plt.title(r"Kaiser window ($\beta$={}),L={}".format(beta, L))
            plt.ylabel("Amplitude")
            plt.xlabel("Sample")
            plt.grid()

            plt.subplot(212)
            A = fft(window, 2048) / (len(window) / 2.0)
            freq = np.linspace(-0.5, 0.5, len(A))
            response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
            plt.plot(freq, response)
            plt.axis([-0.5, 0.5, -120, 0])
            plt.title(r"Frequency response of the Kaiser window ($\beta$={})".format(beta))
            plt.ylabel("Normalized magnitude [dB]")
            plt.xlabel("Normalized frequency [cycles per sample]")
            plt.grid()
            plt.show()

        return window

    @staticmethod
    def get_fft(x_sig: np.ndarray, fs: float, label: str, window=None, display=True):

        print('Computing Fast Fourier Transfor (FFT)...')
        #  Compute the nyquist frequency
        ny_freq = fs / 2

        #  Perform the FFT with or without the windowing
        if not (window is None):
            # Evaluate the fft with the window
            # Multiply the window and the signal, probably we need to split the orginal signal in multiple chunks
            len_sig = len(x_sig)
            len_win = len(window)
            if len_sig < len_win:
                # Padding
                adj_sig = np.zeros(len_win)
                adj_sig[0:len_sig] = x_sig
                num_samples = len(adj_sig)
                ywf = fftshift(fft(np.multiply(adj_sig, window)) / (num_samples / 2))
            elif len_sig > len_win:
                n_chunks_ok = len_sig // len_win
                aligned_idx = len_win * n_chunks_ok
                aligned_values = x_sig[0:aligned_idx]
                chunks = np.split(aligned_values, n_chunks_ok,)
                diff = len_sig - aligned_idx
                last_chunk = np.zeros(len_win)
                last_chunk[0:diff] = x_sig[aligned_idx::]
                chunks.append(last_chunk)
                adj_sig = []
                for chunk in chunks:
                    adj_sig.append(np.multiply(chunk, window))
                adj_sig = np.concatenate(adj_sig, axis=0)
                num_samples = len(adj_sig)
                ywf = fftshift(fft(adj_sig) / (num_samples / 2))
            else:
                num_samples = len(x_sig)
                ywf = fftshift(fft(np.dot(x_sig, window)) / (num_samples / 2))
        else:
            num_samples = len(x_sig)
            ywf = fftshift(fft(x_sig)) / (num_samples / 2)

        # Create the xaxis frequencies
        n = np.arange(num_samples)
        T = num_samples / (ny_freq * 2)
        freq = n / T - ny_freq

        # Display the FFT of the signal
        if display:
            plt.figure(figsize=(10, 6))
            plt.stem(freq, np.abs(ywf), 'b', markerfmt=" ", basefmt="-b")
            plt.xlabel('Freq (Hz)', fontsize=15)
            plt.ylabel('FFT Amplitude |X(freq)|', fontsize=15)
            plt.grid()
            if not (window is None):
                plt.title('{} FFT with kaiser windowing'.format(label), fontsize=15)
            else:
                plt.title('{} FFT'.format(label))
            plt.show()

        return freq, np.abs(ywf)


############################
#      USAGE EXAMPLE       #
############################


if __name__ == "__main__":
    from numpy import cos, sin, pi, arange

    #  Generate the test signal
    sample_rate = 500.0
    nsamples = 400
    t = arange(nsamples) / sample_rate
    x = 10 * cos(2 * pi * 8 * t) + 5 * sin(2 * pi * 20 * t) + 5 * sin(2 * pi * 80 * t)


    # #  Perform the FFT without the signal windowing
    _, _ = Tools.get_fft(x_sig=x, fs=sample_rate, label='test')
    # Perform the FFT with signal windowing using the Kaiser Window
    k_win = Tools.get_kaiser_window(side_lobe_amplitude=60, fs=sample_rate, resolution=2, display=True)
    Tools.get_fft(x_sig=x, fs=sample_rate, window=k_win, label='test')

    ############################
    #       FIR FILTERING      #
    ############################

    #  Create a FIR filter
    FirFilter = FIRLowPass(fs=sample_rate, cutoff_hz=35, ripple_db=60, transition_width=40, id_filter='Test')
    FirFilter.disp_info()

    # Filter the original signal with the FIR filter
    x_fir_filtered = FirFilter.filter(x)
    fir_delay = FirFilter.get_delay()
    fir_M = FirFilter.get_filter_len()

    ############################
    #       BTW FILTERING      #
    ############################

    #  Create a BTW filter
    BtwFilter = BtwLowPass(fs=sample_rate, cutoff_hz=35, order=4, id_filter='Test')
    BtwFilter.disp_info()

    # Filter the original signal with the BTW filter
    x_btw_filtered = BtwFilter.filter(x)

    ############################
    #       NOTCH FILTERING    #
    ############################

    # #  Create a BTW filter
    NtcFilter = NotchIir(f_notch=80.0, fs=sample_rate, q_fact=30.0, id_filter='Notch 80 Hz')
    NtcFilter.disp_info()

    # Filter the original signal with the BTW filter
    x_ntc_filtered = NtcFilter.filter(x)


    ############################
    #        FIGURES           #
    ############################
    plt.figure(20, figsize=(10, 6))
    # Plot the original signal.
    plt.plot(t, x, label='original')
    # Plot the filtered signal
    plt.plot(t, x_fir_filtered, 'g', linewidth=4, label='reliable')

    plt.legend()
    plt.xlabel('t')
    plt.title('FIR filtering Signal')
    plt.grid(True)

    plt.figure(30, figsize=(10, 6))
    # Plot the original signal.
    plt.plot(t, x, label='original')
    # Plot the filtered signal,
    plt.plot(t, x_btw_filtered, 'g', linewidth=4, label='reliable')
    plt.legend()
    plt.xlabel('t')
    plt.title('BTW filtering Signal')
    plt.grid(True)

    plt.figure(40, figsize=(10, 6))
    # Plot the original signal.
    plt.plot(t, x, label='original')
    # Plot the filtered signal,
    plt.plot(t, x_ntc_filtered, 'g', linewidth=4, label='reliable')
    plt.legend()
    plt.xlabel('t')
    plt.title('Ntc filtering Signal')
    plt.grid(True)
