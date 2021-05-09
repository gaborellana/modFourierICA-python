import numpy as np
from utils import *
import glob

class EEGGenerator():
    def __init__(self):
        self.path = 'files/'
        self.Nsrcs = 1509
        self.Nchs = 64
        self.Nsims = 4
        self.minfreq=0.25
        self.maxfreq=20

        self.Fs = 512.             
        self.T = 1/self.Fs
        self.L = 100*self.Fs
        self.t = np.arange(0., self.L/self.Fs - 1/self.Fs, 1/self.Fs)
        self.f = self.Fs*np.linspace(0.,(self.L/2))/self.L
        self.frec = np.arange(0.1, 64, 0.005)
        self.ff = np.arange(self.minfreq, self.maxfreq+0.1, 0.25)
        self.fdom = [3,7,12]
        self.phs = [[0], [0], [0, np.pi/2]]
        self.wid=1.
        self.coh=0.75
        self.ampl=50
        self.alpha = 0.75
        self.Sg = None
        self.loadSigs=False
        self.loadMixNoiseMat=False
        self.loadNoise=True
        self.loadMix=True
        self.SgSrcs = [500, 250, 1000, 750]
        
    def calculate(self):
        if self.loadSigs:
            self.Sg = np.load(self.path + 'Sg.npy')
        else:
            self._sigGen()
        if self.loadMixNoiseMat:
            self.NoiseMix = np.load(self.path + 'NoiseMix.npy')
        else:
            if self.loadNoise:
                self._noiseLoad(self.path)
            else:
                self._noiseGen(chs=10)
            self._noiseMixing()
        for i,src in enumerate(self.SgSrcs):
            self.NoiseMix[src,:]= self.NoiseMix[src,:]+self.Sg[i,:]*1.
        if self.loadMix:
            self.mixing = np.load(self.path + 'Mix.npy')
        else:
            self._mixGen()
        
    def getEEG(self):
        return self.mixing @ self.NoiseMix
    
    def getSources(self):
        return self.NoiseMix
        
    def _sigGen(self):
        for f in range(len(self.fdom)):
            aux, _ = signalGen(t=self.t, fdom=self.fdom[f], wid=self.wid, 
                               ph=np.array(self.phs[f]), coh=self.coh)
            aux=aux*self.ampl*self.fdom[f]**(-self.alpha)
            if self.Sg is None:
                self.Sg=aux
            else:
                self.Sg = np.concatenate((self.Sg,aux),0)
        with open(self.path + 'Sg.npy', 'wb') as f:
            np.save(f, self.Sg)
                
    def _noiseLoad(self, noise_path='./'):
        noiseList = sorted(glob.glob(noise_path + 'noise_*.npy'))
        self.Nns = len(noiseList)
        self.noise = np.zeros((self.Nns, self.Sg.shape[1]))
        for ns in range(self.Nns):
            self.noise[ns,:]=np.load(noiseList[ns])
            
    def _noiseGen(self, noiseChans=10):
        self.Nns = noiseChans
        print('Noise channels will be generated. It can take a while.')
        for i in range(0,10):
            print('Generating noise channel ' + str(i))
            Ns = noiseGen(t,frec,alpha)
            nstr = str(i)
            if len(nstr)==1:
                nstr = '00'+nstr
            elif len(nstr)==2:
                nstr = '0'+nstr
            with open(self.path + 'noise_'+nstr+'.npy', 'wb') as f:
                np.save(f, Ns)
        _self.noiseLoad()
        
    def _noiseMixing(self):
        self.NoiseMix = np.zeros((self.Nsrcs,self.Sg.shape[1]))
        chunk_len=1000
        chunks = np.zeros(int(np.ceil(self.Sg.shape[1]/chunk_len))+1).astype(int)
        chunks[:-1] = np.arange(0,self.Sg.shape[1],chunk_len)
        chunks[-1] = self.Sg.shape[1]
        MixMatOld = normMatFun(np.random.rand(self.Nns,self.Nsrcs)*2-1)
        for c in range(len(chunks)-1):
            MixMatNew = normMatFun(np.random.rand(self.Nns,self.Nsrcs)*2-1) # 250x15002
            step = (MixMatNew-MixMatOld)/(chunks[c+1]-chunks[c])
            del MixMatNew
            for j in range(chunks[c], chunks[c+1]):
                MixMatOld=MixMatOld+step
                Y = self.noise[:,j]
                self.NoiseMix[:,j] = normMatFun(MixMatOld).T @ Y
        with open(self.path + 'NoiseMix.npy', 'wb') as f:
            np.save(f, self.NoiseMix)
        
    def _mixGen(self):
        while 1:
            self.mixing = np.random.rand(self.Nchs,self.Nsrcs)*2-1
            for r in range(Nchs):
                self.mixing[r,:] = self.mixing[r,:]/np.sum(np.abs(self.mixing[r,:]))
            if np.max(np.abs(self.mixing))<1:
                break

