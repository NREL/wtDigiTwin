from .kalman import *
import numpy as np

class KalmanFilter(object):
    def __init__(self,sX0,sXa,sU,sY):
        self.sX0 = sX0
        self.sXa = sXa
        self.sU  = sU
        self.sY  = sY

        #  State vector is States and Augmented states
        self.sX=np.concatenate((self.sX0,self.sXa))

        # --- Defining index map for convenience
        self.iX={lab: i   for i,lab in enumerate(self.sX)}
        self.iY={lab: i   for i,lab in enumerate(self.sY)}

    @property
    def nX(self):
        return len(self.sX)

    @property
    def nY(self):
        return len(self.sY)

    @property
    def nU(self):
        return len(self.sU)

    @property
    def nP(self):
        return len(self.sXa)

    @property
    def nX0(self):
        return len(self.sX0)

    def __repr__(self):
        s=''
        s+='<kalman.KalmanFilter object> \n'
        s+='  sX0 : {} \n'.format(self.sX)
        s+='  sX0 : {} \n'.format(self.sX0)
        s+='  sX1 : {} \n'.format(self.sXa)
        s+='  sU  : {} \n'.format(self.sU)
        s+='  sY  : {} \n'.format(self.sY)
        return s


    def setMat(self, Xx, Xu, Yx, Yu):
        # --- 
        self.Xx, self.Xu, self.Yx, self.Yu= EmptyStateMat(self.nX,self.nU,self.nY)

        if Xx.shape != self.Xx.shape:
            raise Exception('Shape of Xx ({}) not compatible with KF Xx shape ({}) '.format(Xx.shape, self.Xx.shape))
        if Xu.shape != self.Xu.shape:
            raise Exception('Shape of Xu ({}) not compatible with KF Xu shape ({}) '.format(Xu.shape, self.Xu.shape))
        if Yx.shape != self.Yx.shape:
            raise Exception('Shape of Yx ({}) not compatible with KF Yx shape ({}) '.format(Yx.shape, self.Yx.shape))
        if Yu.shape != self.Yu.shape:
            raise Exception('Shape of Yu ({}) not compatible with KF Yu shape ({}) '.format(Yu.shape, self.Yu.shape))
        self.Xx=Xx
        self.Xu=Xu
        self.Yx=Yx
        self.Yu=Yu

    def discretize(self,dt,method='exponential'):
        self.dt=dt
        self.Xxd,self.Xud = KFDiscretize(self.Xx, self.Xu, dt, method=method)

    def estimateTimeStep(self,u,y,x,P,Q,R):
        """
        OUTPUTS:
          z1: States at time n
          P1: Process covariance at time n
          Kk: Kalman gain
        """
        return EstimateKFTimeStep(u,y,x,self.Xxd,self.Xud,self.Yx,self.Yu,P,Q,R)

    def covariancesFromSig(self):
        if not hasattr(self,'sigX'):
            raise Exception('Set `sigX` before calling `covariancesFromSig` (e.g. `sigmasFromClean`)')
        if not hasattr(self,'sigY'):
            raise Exception('Set `sigY` before calling `covariancesFromSig` (e.g. `sigmasFromClean`)')
        P = np.eye(self.nX)
        Q = np.diag([self.sigX[lab]**2 for lab in self.sX])
        R = np.diag([self.sigY[lab]**2 for lab in self.sY])
        return P,Q,R



    # --------------------------------------------------------------------------------}
    # --- TIME, Optional convenient methods if a time vector is already available
    # --------------------------------------------------------------------------------{
    def setTimeVec(self,time):
        self.time=time

    @property
    def nt(self):
        return len(self.time)

    def setCleanValues(self,df,ColMap=None):
        # --- Defining "clean" values 
        self.X_clean = np.zeros((self.nX,self.nt))
        self.Y_clean = np.zeros((self.nY,self.nt))
        self.U_clean = np.zeros((self.nU,self.nt))
        for i,lab in enumerate(self.sX):
            self.X_clean[i,:]=df[ColMap[lab]]
        for i,lab in enumerate(self.sY):
            self.Y_clean[i,:]=df[ColMap[lab]]
        for i,lab in enumerate(self.sU):
            self.U_clean[i,:] =df[ColMap[lab]]

    def setY(self,df,ColMap=None):
        for i,lab in enumerate(self.sY):
            self.Y[i,:]=df[ColMap[lab]]

    def initTimeStorage(self):
        self.X_hat = np.zeros((self.nX,self.nt))
        self.Y_hat = np.zeros((self.nY,self.nt))
        self.Y     = np.zeros((self.nY,self.nt))
    
    # TODO use property or dict syntax
    def get_vY(self,lab):
        return self.Y[self.iY[lab],:]
    def set_vY(self,lab, val ):
        self.Y[self.iY[lab],:]=val

    def get_vX_hat(self,lab):
        return self.X_hat[self.iX[lab],:]
    def set_vX_hat(self,lab, val ):
        self.X_hat[self.iX[lab],:]=val

    def get_Y(self,lab,it):
        return self.Y[self.iY[lab],it]
    def set_Y(self,lab, val ):
        self.Y[self.iY[lab],it]=val

    def get_X_hat(self,lab,it):
        return self.X_hat[self.iX[lab],it]
    def set_X_hat(self,lab, val ):
        self.X_hat[self.iX[lab],it]=val



    def initFromClean(self):
        x = self.X_clean[:,0]
        # x = np.zeros(nX)
        self.X_hat[:,0] = x
        self.Y_hat[:,0] = self.Y_clean[:,0]
        return x

    def initZero(self):
        return np.zeros(self.nX)


    def setYFromClean(self,NoiseRFactor=None,y_bias=None,R=None):
        if y_bias is None:
            y_bias = np.zeros(self.nY)

        if NoiseRFactor is not None:
            Ey = np.sqrt(R)*NoiseRFactor

        for it in range(0,self.nt-1):    
            self.Y[:,it+1] = self.Y_clean[:,it] + np.dot(Ey,np.random.randn(self.nY,1)).ravel() + y_bias

    def sigmasFromClean(self,factor=1):
        sigX   = dict()
        for iX,lab in enumerate(self.sX):
            std = np.std(self.X_clean[iX,:])
            if std==0:
                res=1
            else:
                res=10**(np.floor(np.log10(std))-1)
            sigX[lab]=np.floor(std/res)*res  * factor
        sigY   = dict()
        for iY,lab in enumerate(self.sY):
            std = np.std(self.Y_clean[iY,:])
            if std==0:
                res=1
            else:
                res=10**(np.floor(np.log10(std))-1)
            sigY[lab]=np.floor(std/res)*res * factor
        self.sigX=sigX
        self.sigY=sigY
        return sigX,sigY

    def print_sigmas(self,sigX_c=None,sigY_c=None):
        if sigX_c is not None:
            print('Sigma X            to be used     from inputs')
            for k,v in self.sigX.items():
                print('Sigma {:10s}: {:12.3f}  {:12.3f}'.format(k,v,sigX_c[k]))
        else:
            print('Sigma X            to be used')
            for k,v in self.sigX.items():
                print('Sigma {:10s}: {:12.3f}'.format(k,v))

        if sigY_c is not None:
            print('Sigma Y            to be used     from inputs')
            for k,v in self.sigY.items():
                print('Sigma {:10s}: {:12.3f}  {:12.3f}'.format(k,v,sigY_c[k]))
        else:
            print('Sigma Y            to be used')
            for k,v in self.sigY.items():
                print('Sigma {:10s}: {:12.3f}'.format(k,v))


    # --------------------------------------------------------------------------------}
    # --- Plot functions 
    # --------------------------------------------------------------------------------{
    def plot_Y(KF,fig=None):
        import matplotlib
        import matplotlib.pyplot as plt
        # --- Compare measurements
        cmap = matplotlib.cm.get_cmap('viridis')
        COLRS = [(cmap(v)[0],cmap(v)[1],cmap(v)[2]) for v in np.linspace(0,1,3+1)]
        if fig is None:
            fig=plt.figure()
        for j,s in enumerate(KF.sY):
            ax=fig.add_subplot(KF.nY,1,j+1)
            ax.plot(KF.time,KF.Y_clean[j,:],''  ,  color=COLRS[0] ,label='Clean')
            ax.plot(KF.time,KF.Y[j      ,:],'-.',  color=COLRS[2] ,label='Noisy')
            ax.plot(KF.time,KF.Y_hat[j  ,:],'--' , color=COLRS[1],label='Estimate')
            ax.set_ylabel(s)
        ax.set_title('Measurements Y')

    def plot_X(KF,fig=None):
        import matplotlib
        import matplotlib.pyplot as plt
        # --- Compare States
        cmap = matplotlib.cm.get_cmap('viridis')
        COLRS = [(cmap(v)[0],cmap(v)[1],cmap(v)[2]) for v in np.linspace(0,1,3+1)]
        if fig is None:
            fig=plt.figure()
        for j,s in enumerate(KF.sX):
            ax=fig.add_subplot(KF.nX,1,j+1)
            ax.plot(KF.time,KF.X_clean[j,:],''  , color=COLRS[0],label='Clean')
            ax.plot(KF.time,KF.X_hat  [j,:],'--', color=COLRS[1],label='Estimate')
            ax.set_ylabel(s)
        ax.set_title('States X')

