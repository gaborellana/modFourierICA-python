import numpy as np
from scipy.fft import fft
from scipy.signal.windows import blackman
import scipy.stats as stats


def signalGen(t, fdom, wid, ph=None, coh=None):
    if ph is None:
        ph = (0)
        N=1
    else:
        N=len(ph)
    if coh is None:
        coh=1
    Ns = np.zeros(len(t))
    step = t[1]-t[0]
    w = 2*np.pi*fdom*step
    w1 = 2*np.pi*wid*step
    ## probs change of state
    prob01 = 0.001#step/5 *10
    prob10 = 0.1#1/fdom
    state = 0
    rnd3 = 0
    Rnd3 = []
    Rnd3.append(rnd3)
    flag = 1
    cycleind = []
    for k in range(len(t)-1):
        k=k+1
        if state==0:
            rnd1 = np.random.rand(1)[0]
            if rnd1<prob01:
                state = 1
                cycleind.append(k)
                flag=0
            else:
                Ns[k] = Ns[k-1]
        if state==1:
            Ns[k] = Ns[k-1]+(w+rnd3)
            cond1=np.sin(Ns[k])*np.sin(Ns[k-1])<=0
            cond2=np.sin(Ns[k-1])<np.sin(Ns[k])
            if cond1 and cond2 and flag:
                cycleind.append(k)
                rnd2 = np.random.rand(1)[0]
                if rnd2<prob10:
                    state = 0
                    Ns[k] = np.round(Ns[k]/(2*np.pi))*2*np.pi+w*0.000001
                rnd3 = stats.semicircular.rvs(size=1)*w1
                Rnd3.append(rnd3)
        if flag==0: flag=1
    Sg = np.zeros((N,len(t)))
    Sg[0,:] = np.sin(Ns)
    if N>1:    
        ph = ph-ph[0]
        for n in range(1,N):
            dephs = int(np.round(ph[n]/w))
            aux = np.copy(Sg[0,:])
            for c in range(len(cycleind)-1):
                rrnd = np.random.rand(1)
                if rrnd<(1-coh):
                    aux[cycleind[c]:cycleind[c+1]]=0
            if dephs==0:
                Sg[n,:] = aux[:]
            else:
                Sg[n,dephs:] = aux[:-dephs]
    return Sg, Ns


def noiseGen(t,frecs,alpha):
    Noise = np.zeros(len(t))
    rcoef = 0.01
    ncoeff=0.03
    bias = 0.2
    stp = t[1]-t[0]
    for f in frecs:
        P = np.zeros(Noise.shape)
        for k in range(1,len(t)):
            P[k] = ncoeff*(np.random.randn(1)+bias) -rcoef*P[k-1]
            
        Ns = np.zeros(P.shape)
        Ns[0] = np.random.rand(1)*2*np.pi
        w = 2*np.pi*f*stp
        
        for i in range(1,len(t)):
            do = w*(P[i]+1)
            Ns[i] = Ns[i-1]+do
        
        A = np.sin(Ns)
        Noise = Noise+A*(f**-alpha)
    return Noise


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

