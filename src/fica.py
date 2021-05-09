import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal.windows import blackman
from utils import *
from eeggen import *

class fourierICA():
    def __init__(self, eeggen, pcadim):
        self.showprogress = True
        self.seed = -1
        self.lambd = 1
        self.conveps = 1e-7
        self.complexmixing = 0
        self.winlen_sec = 1
        self.overlapfactor = 8
        self.components = eeggen.Nchs #64
        self.removeoutliers = 1
        self.maxiter = np.max(np.array([40*pcadim, 2000]))
        #self.fcnoutliers = []
        self.zerotolerance = self.conveps
        self.hammingdata = True
        self.pcadim = pcadim
        self.eeggen = eeggen
        self.normSpec = True
    
    def processEEGGenerator(self):
        eegdata = self.eeggen.getEEG()
        self.X, params = toSpec(eegdata, self.eeggen.Fs, self.eeggen.minfreq, self.eeggen.maxfreq, 4)
        self.fftsize, self.wins, self.chans,_,_ = params
        if self.normSpec:
            self._normalizeSpec()
        if self.removeoutliers:
            self._removeOutliers()
        self._pca()
        self._complexFastICA()
        print('get results with X = fica.getResults()')

        
    def getResults(self):
        #compute objective function for each component
        objective_=-np.mean(np.log(self.lambd+np.abs(self.S*np.conj(self.S))),1)
        #sort components using the objective
        objective = -np.sort(-objective_)
        componentorder = np.argsort(-objective_)
        W=self.W[componentorder,:]
        A=self.A[:,componentorder]
        S=self.S[componentorder,:]
        #Compute mixing and demixing matrix in original channel space
        W_orig=W @ self.whiteMat   #Spatial filters
        A_orig=self.dewhiteMat @ A #Spatial patterns
        #Independent components. If real mixing matrix, these are STFT's of the sources.
        S_FT=np.transpose(np.reshape(self.S,(self.components,self.fftsize,self.wins), order='F'),axes=[1,2,0])
        self._plotSpecs(S_FT)
        if self.normSpec:
            return (S_FT,A_orig,W_orig,self.normMat)
        else:
            return (S_FT,A_orig,W_orig)
        
    def _normalizeSpec(self):
        mn = np.mean(np.abs(self.X), (0,1), keepdims=True) #normalize channels
        self.X = self.X/mn
        self.normMat = np.median(np.abs(self.X), (1,2), keepdims=True)
        self.X = self.X/self.normMat
    
    def _removeOutliers(self):
        print('Outlier removal:')
        lognorms=np.log(np.sum(np.abs(self.X * np.conj(self.X)),-1))
        outlierthreshold=np.mean(lognorms)+3*np.std(lognorms)
        outlierindices=(lognorms>outlierthreshold).nonzero()[0]
        print(' removed ',len(outlierindices),' windows\n');
        self.X[:,outlierindices]=0
        self.Xmat_c=self.X.reshape((self.fftsize*self.wins, self.chans), order='F') # (64, 3840)
        del self.X
        self.N=self.fftsize*self.wins
        
    def _pca(self):
        Xmat_c_mv=np.mean(self.Xmat_c,-1,keepdims=True)
        self.Xmat_c=self.Xmat_c-Xmat_c_mv # (64, 3840)
        del Xmat_c_mv
        covmat=np.cov(self.Xmat_c.T) #Do PCA (on matrix Xmat_c)
        if not self.complexmixing:
            covmat=np.real(covmat)
        else:
            covmat=np.conj(covmat)
        eVa, eVe = la.eigh(covmat)
        d = -np.sort(-eVa)
        order = np.argsort(-eVa)
        del covmat, eVa
    
        # Checks for negative eigenvalues
        if np.sum(d[:self.pcadim]<0):
            print('Negative eigenvalues! Reducing PCA and ICA dimension...')

        # Check for eigenvalues near zero (relative to the maximum eigenvalue)
        zeroeigval=np.sum((d[:self.pcadim]/d[0])<self.zerotolerance)
        # Adjust dimensions if necessary (because zero eigenvalues were found)
        self.pcadim=self.pcadim-zeroeigval

        if self.pcadim<self.components:
            self.components=self.pcadim
        if zeroeigval:
            print('PCA dimension is ' + str(self.pcadim))
            print('ICA dimension is ' + str(self.components))

        # Construct whitening and dewhitening matrices
        dsqrt = np.sqrt(d[:self.pcadim])
        dsqrtinv = 1/dsqrt
        eVe = eVe[:,order[:self.pcadim]]
        self.whiteMat = np.diag(dsqrtinv) @ eVe.conj().T # (40, 3840) 
        self.dewhiteMat = eVe @ np.diag(dsqrt) # (3840, 40) 
        del d, order, zeroeigval, dsqrt, dsqrtinv, eVe
        # Reduce dimensions and whiten data. |Zmat_c| is the main input for the iterative algorithm
        self.Zmat_c=np.matmul(self.whiteMat,self.Xmat_c.T)
        del self.Xmat_c
        
    def _complexFastICA(self):
        Zmat_c_tr=self.Zmat_c.conj().T # Also used in the fixed-point iteration.
        # Sets the random number generator
        if self.seed==-1:
            numrandn=np.random.rand(1)
        else:
            print("asdads")
            #stream=RandStream('mrg32k3a','seed',19)
            #numrandn=@(x,y)randn(stream,x,y)
        # Initial point, make it imaginary and unitary
        if self.complexmixing:
            W_old=np.random.randn(self.components,self.pcadim) + np.random.randn(self.components,self.pcadim) * 1j
        else:
            W_old=np.random.randn(self.components,self.pcadim)
        W_old=la.sqrtm(la.inv(W_old @ W_old.conj().T)) @ W_old

        # Iteration starts here
        for it in range(self.maxiter):
            #Compute outputs, note lack of conjugate
            Y = W_old @ self.Zmat_c
            #Computing nonlinearities
            Y2=np.abs(Y*np.conj(Y))
            gY2 = 1/(self.lambd + Y2)
            dmv=self.lambd*np.sum(gY2**2,1)
            #Fixed-point iteration #(Y.*gY2)*Zmat_c_tr-diag(dmv)*W_old;
            W_new=(Y*gY2) @ Zmat_c_tr - np.diag(dmv) @ W_old
            #In case we want to restrict W to be real-valued, do it here:
            if self.complexmixing: 
                W=np.copy(W_new)
            else:
                W=np.copy(np.real(W_new))
            #Make unitary
            W = la.sqrtm(la.inv(W @ W.conj().T)) @ W 
            #check if converged
            convcriterion=1-np.sum(np.abs(np.sum(W*np.conj(W_old),1)))/self.components
            if convcriterion<self.conveps:
                break
            if it%50==0:
                print(convcriterion)
            # store old value
            W_old=np.copy(W)
        del Y, Y2, gY2, dmv, W_old, W_new, Zmat_c_tr
        self.W = W
        # Compute mixing matrix (in whitened space)
        self.A=self.W.conj().T
        # Compute source signal estimates
        self.S=self.W @ self.Zmat_c
        if convcriterion>self.conveps:
            print('\nFailed to converge, results may be wrong!\n')
        else:
            print('\nConverged.\n')
            
    def _plotSpecs(self, S_FT):
        if self.normSpec:
            S_FT = S_FT*self.normMat
        specs = np.mean(np.abs(S_FT),1)
        specs = specs/np.mean(np.abs(specs), 0,keepdims=True)
        plt.plot(self.eeggen.ff, np.log(specs+0.0001))
        plt.show()
    
def normMatFun(MixMat, dm=0):
    S = np.sum(np.abs(MixMat),dm, keepdims=True)
    return MixMat/S





def toSpec(data,sf,minfreq,maxfreq,winlen_sec):
    # Function based on fourierICA.m
    # https://www.cs.helsinki.fi/group/neuroinf/code/fourierica/html/fourierica.html
    overlapfactor = 8
    hamm=True
    chans,T=data.shape 
    winsize=int(np.floor(winlen_sec*sf))
    wininterval=int(np.ceil(winsize/overlapfactor))
    numwins=int(np.floor((1.*T-winsize)/wininterval+1))

    startfftind=int(np.floor(minfreq*winlen_sec)) #compute frequency indices (for the STFT)
    if startfftind<0:
        print('minfreq must be positive')
        perro=gato
    
    endfftind=int(np.floor(maxfreq*winlen_sec))+1
    nyquistfreq=int(np.floor(winsize/2))
    if endfftind>nyquistfreq:
        print('maxfreq must be less than the Nyquist frequency')
        perro=gato
        
    fftsize=int(endfftind-startfftind)
    # Initialization of tensor X, which is the main data matrix input to the code which follows.
    X=np.zeros((fftsize,numwins,chans)).astype(np.complex64)    
    window=np.array([0,winsize]) # Define window initial limits
    if hamm:
        hammwin=blackman(winsize, False) # Construct Hamming window if necessary 
    # Short-time Fourier transform (window sampling + fft)
    for j in range(numwins):
        datawin=data[:,window[0]:window[1]] # Extract data window
        if hamm:  
            datawin=datawin @ np.diag(hammwin) # Multiply by a Hamming window if necessary
        datawin_ft=fft(datawin) # Do FFT
        X[:,j,:]=(datawin_ft[:,startfftind:endfftind]).T
        window=window+wininterval # New data window interval
    params = (fftsize, numwins, chans,wininterval,winsize)    
    return X, params

        