from typing import Any
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

_PATH = 'F:\MASTER\proiect TB\database\Balabaneanu_Madalina_2_r.npy'


class SignalProcessing():
    def __init__(self, PATH) -> None:
        self.signal = np.load(PATH)
        self.numChannels = self.signal.shape[0]
        self.channels = [s for s in self.signal]
        self.segmentSize = int(self.signal.shape[1] / 2)
        self.activeMuscle = np.empty([int(self.segmentSize)])
        self.fatigueMuscle = np.empty([int(self.segmentSize)])
        self.samplingFreq = 512
        self.frameSize = 768 # represents 90sec of a signal with fs=512
        self.hopLength = 512 # 30sec overlapping (hopping with 60sec after each 90sec frame)
        self.zeroCenterSignal()

    def __call__(self) -> None:
        self.signalSplit()
        return self.featureExtraction(self.activeMuscle), self.featureExtraction(self.fatigueMuscle)
        
    def zeroCenterSignal(self):
        self.signal = self.signal.astype(np.int8)
        for idx in range(0, self.signal.shape[0]):
            self.signal[idx] = np.subtract(self.signal[idx], 128, dtype='int8')
  

    def signalSplit(self) -> None:
        for idx, channel in enumerate(self.signal):
            activeMuscleChannel = np.array(channel[:self.segmentSize])
            fatigueMuscleChannel = np.array(channel[self.segmentSize:])
            if idx == 0:
                self.activeMuscle = activeMuscleChannel
                self.fatigueMuscle = fatigueMuscleChannel
            else:
                self.activeMuscle = np.vstack((self.activeMuscle, activeMuscleChannel))
                self.fatigueMuscle  = np.vstack((self.fatigueMuscle , fatigueMuscleChannel))
    
    def meanAbsoluteValue(self, samples) -> Any:
        _MAV = []
        for idx in range(0, len(samples), self.hopLength):
            mavCurrentFrame = np.mean(sum(np.abs(samples[idx:idx+self.frameSize])))
            _MAV.append(mavCurrentFrame)
        _MAVcopy = _MAV
        for idx in range(0, len(_MAVcopy)):
            _MAV[idx] = (_MAVcopy[idx] - min(_MAVcopy)) / (max(_MAVcopy) - min(_MAVcopy))

        return np.array(_MAV)

    def meanFrequencySpec(self, samples) -> Any:
        ampl = np.abs(np.fft.fft(samples))
        freq = np.fft.fftfreq(ampl.shape[0], d=1.0/self.samplingFreq)
        _MNF = 0
        # for idx in range(0, freq.shape[0]):
        #     freq[idx] = (freq[idx] - min(freq)) / (max(freq) - min(freq))
        # for idx in range(0, ampl.shape[0]):
        #     ampl[idx] = (ampl[idx] - min(ampl)) / (max(ampl) - min(ampl))
        for idx in range(0, freq.shape[0]):
            _MNF += freq[idx] * ampl[idx]
        _MNF = _MNF / sum(ampl)

        return _MNF

    def iESMG(self, samples) -> Any:
        _iESMG = []
        for idx in range(0, len(samples), self.hopLength):
            iESMGCurrentFrame =  np.sqrt(sum(np.abs(samples[idx:idx+self.frameSize])))
            _iESMG.append(iESMGCurrentFrame)
        iESMGcopy = _iESMG
        for idx in range(0, len(iESMGcopy)):
            _iESMG[idx] = (iESMGcopy[idx] - min(iESMGcopy)) / (max(iESMGcopy) - min(iESMGcopy))        
        return _iESMG
    
    def zeroCrossingRate(self, samples) -> Any:
        _THRESHOLD = 0
        _ZCR = []
        for index in range(0, samples.shape[0], self.hopLength):
            zeroCrossingRate = 0
            for idx, sample in enumerate(samples[index:index+self.frameSize-1]):
                if (sample > _THRESHOLD and samples[idx + 1] < _THRESHOLD) or (
                        sample < _THRESHOLD and samples[idx + 1] > _THRESHOLD):
                    zeroCrossingRate += 1
            _ZCR.append(zeroCrossingRate)
        _ZCRcopy = _ZCR
        for idx in range(0, len(_ZCR)):
            _ZCR[idx] = (_ZCRcopy[idx] - min(_ZCRcopy)) / (max(_ZCRcopy) - min(_ZCRcopy))
            
        return np.array(_ZCR)
    
    def featureVector(self, *args) -> Any:
        featureVec = args[0]
        for arg in args[1:]:
            featureVec = np.hstack((featureVec, arg))

        return featureVec


    def featureExtraction(self, inputSignal) -> Any:
        for idx, channel in enumerate(inputSignal):
            zcr = self.zeroCrossingRate(channel)
            mav = self.meanAbsoluteValue(channel)
            iesmg = self.iESMG(channel)
            if idx == 0:
                featureArray = self.featureVector(zcr, mav, iesmg)
            else:
                featureArray = np.vstack((featureArray, self.featureVector(zcr, mav, iesmg)))
        return featureArray
    
    def specShow(self):
        
        for idx, channel in enumerate(self.signal):
            # f, t, Sxx = signal.spectrogram(channel, self.samplingFreq)
            # plt.pcolormesh(t, f, Sxx, shading='gouraud')
            plt.plot(channel)
            plt.title(f'Channel {idx+1}')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()

class TestingTool():
    def __init__(self) -> None:
        self.features = SignalProcessing(_PATH)

    def __call__(self) -> Any:
        return self.features.specShow()


if __name__=='__main__':
    test = TestingTool()
    test()
