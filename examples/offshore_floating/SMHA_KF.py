""" 

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from physical_models import get_physical_model
from FTNSLin import KalmanFilterFTNSLin # TODO

from welib.essentials import *

# Big welib mess... # TODO simplify me
from welib.kalman.kalman import EmptyStateMat, EmptyStateDF
from welib.yams.models.packman import IMUjacobian
from welib.fast.extract import extractIMUAccFromLinFile, mainLinInputs

from welib.fast.linmodel import DEFAULT_COL_MAP_LIN
from welib.fast.linmodel import FASTLinModelFTNSB

from welib.fast.tools.lin import subMat, matSimpleStateLabels, matToSIunits, renameList
from welib.weio.fast_linearization_file import FASTLinearizationFile


np.set_printoptions(linewidth=300, precision=2)
pd.set_option('display.max_rows', 50, 'display.max_columns', 50,'display.width', 400, 'display.precision',2)

# ---- Script parameters
tRange     = [0,100]
tRange     = [0,700]
nUnderSamp = 0
NoiseRFactor=0.0
bFilterAcc=False  # FILTER ACC IMPROVES SPECTRA BUT INCREASE REL ERR OF My
nFilt=15


# --- Tuning
zero_threshold=1e-9 # Watch out for 1/J ~ e-8
sThrust='NacFxN1_[N]'
sThrust='Thrust'
kThrust=0.3 # Tuning factor for thrust measured acceleration feedback
kThrustA=1. # Tuning factor for thrust state accelerations
kIMUz_z=3   # Tuning factor z into IMUz
kSigQaero=50 # Tuning of covariance for aero torque
fullColumns=False #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< IMPORTANT
fullColumns=True #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< IMPORTANT


# --- YAMS Inputs
YAMS=False
modelName=None


# --- Tetra Spar
DOFs  = 'x,y,z,phi_x,phi_y,phi_z,q_FA1,psi'
sMeas = 'x,y,phi_x,phi_y,dpsi,NcIMUAx,NcIMUAy,NcIMUAz,Qgen'
# sMeas = 'dpsi,NcIMUAx,NcIMUAy,NcIMUAz,Qgen' # Challenging
sInp  = 'Qgen,pitch' 
# sStor = 'WS,Qaero,dpsi,pitch,phi_y,WS0'
sStor = ''
sAug = 'Qaero'

# --------------------------------------------------------------------------------}
# --- TetraSpar 
# --------------------------------------------------------------------------------{
WSE_pklFilename = None # TODO NREL 5MW

# --- 
modelName = 'F001000T0N0S0_fnd_hydroO';

# --- Turbulent wind regular wave NREL5MW Spar
fstLin = '../../_data/Spar_lin/Main.fst';  labelLin='NoAeroLin'
fstSim = '../../_data/Spar_sim/Main.fst'; 
kSigQaero=1 # Tuning of covariance for aero torque


# --- Kalman filter model
class KalmanModel():
    """
    Open up a linear physical model, convert it to an augmented system

    Main Data:
        KM.sStates : list of states
        KM.sAug    : list of augmented states
        KM.sMeas   : list of measurements
        KM.sInp    : list of inputs
        KM.sStor   : list of additional storage
        KM.Xx = Xx
        KM.Xu = Xu
        KM.Yx = Yx
        KM.Yu = Yu
    """
    def __init__(KM, modelName, fstSim, fstLin, usePickle=True, DOFs='', sMeas='', sInp='', sAug='', sStor=''):
        KM.StateModel=''


        # Col MAP for OpenFAST OutFile
        sIMU=['NcIMUAx','NcIMUAy','NcIMUAz']
        sIMU2=['NcIMUAx','NcIMUAy','NcIMUAz','NcIMUVx','NcIMUVy','NcIMUVz']
        KM.ColMap={
          ' x      ' : ' PtfmSurge_[m]                   '              ,
          ' y      ' : ' PtfmSway_[m]                   '               ,
          ' z      ' : ' PtfmHeave_[m]                   '              ,
          ' phi_x  ' : ' {PtfmRoll_[deg]}   * np.pi/180               ' , # SI [deg] -> [rad]
          ' phi_y  ' : ' {PtfmPitch_[deg]}  * np.pi/180                ', # SI [deg] -> [rad]
          ' phi_z  ' : ' {PtfmYaw_[deg]}    * np.pi/180              '  , # SI [deg] -> [rad]
          #' q_FA1  ' : ' TTDspFA_[m]                   '                ,
          ' q_FA1  ' : ' Q_TFA1_[m]                   '                ,
          ' psi    ' : ' {Azimuth_[deg]} * np.pi/180   '                , # SI [deg] -> [rad]
          ' dq_FA1 ' : ' QD_TFA1_[m/s]               '                ,
          ' dpsi  ' : ' {RotSpeed_[rpm]} * 2*np.pi/60 '                , # SI [rpm] -> [rad/s]
          ' dx     ' : ' QD_Sg_[m/s]               '              ,
          ' dy     ' : ' QD_Sw_[m/s]                '               ,
          ' dz     ' : ' QD_Hv_[m/s]               '              ,
          ' dphi_x ' : ' QD_R_[rad/s]                             ' ,
          ' dphi_y ' : ' QD_P_[rad/s]                             ',
          ' dphi_z ' : ' QD_Y_[rad/s]                             '  ,
          ' Thrust ' : ' RtFldFxh_[N]                 '                ,
          ' Qaero  ' : ' RtFldMxh_[N-m]               '                ,
          ' Qgen   ' : ' {GenTq_[kN-m]}  *1000         '             , # [kNm] -> [Nm]
#           ' Qgen   ' : ' 97*{GenTq_[kN-m]}  *1000         '             , # [kNm] -> [Nm]
          ' WS     ' : ' RtVAvgxh_[m/s]                '                ,
          ' pitch  ' : ' {BldPitch1_[deg]} * np.pi/180 '                , # SI [deg]->[rad]
          ' NcIMUAx ' : ' NcIMUTAxs_[m/s^2]             ',
          ' NcIMUAy ' : ' NcIMUTAys_[m/s^2]             ',
          ' NcIMUAz ' : ' NcIMUTAzs_[m/s^2]             ',
          ' NcIMUVx ' : ' NcIMUTVxs_[m/s]             ',
          ' NcIMUVy ' : ' NcIMUTVys_[m/s]             ',
          ' NcIMUVz ' : ' NcIMUTVzs_[m/s]             ',
          }


        ColMapLinFile = DEFAULT_COL_MAP_LIN


        # --------------------------------------------------------------------------------}
        # ---  Linear Physical model
        # --------------------------------------------------------------------------------{
        if YAMS:
            # --- YAMS
            sysLI, sim = get_physical_model(modelName, fstSim, qop=None, qdop=None, usePickle=usePickle, noBlin=True)
            sX0= list(sim.WT.DOFname)
            sX = sX0 + ['d'+sx for sx in sX0]
            A_YAMS = pd.DataFrame(data=sysLI.A, index=sX, columns=sX)
            #dq  = ((np.max(dfFS[sq]) -np.min(dfFS[sq]))/100).values
            #dqd = ((np.max(dfFS[sqd])-np.min(dfFS[sqd]))/100).values
            #Kacc_fd, Cacc_fd = IMUjacobian(pkg, q0, qd0, p, 'finiteDifferences', dq, dqd)
            uop = sim.uop
            u=dict()
            for key in sim.pkg.info()['su']:
                u[key]= lambda t,q=None,qd=None: 0
            Kacc, Cacc, acc0 = IMUjacobian(sim.pkg, qop=None, qd0=None, p=sim.p, u=u, uop=uop, method='packageJacobians', sDOFs=sX0)
            CIMU_YAMS = pd.concat((Kacc,Cacc),axis=1)
            CIMU_YAMS.index=sIMU
            #B = matToSIunits(B, 'B')
            A=A_YAMS

        # OpenFAST
        if not YAMS:
            #linFile=os.path.splitext(fstFilename)[0]+'.1.lin'
            #if not os.path.exists(linFile):
            #    raise Exception('Cannot load lin file : {}'.format(linFile))
            #print('Loading lin file: ', linFile)
            #lin = FASTLinearizationFile(linFile)
            #linDFs = lin.toDataFrame()

            #A_OF = matSimpleStateLabels(linDFs['A'])
            #CIMU_OF, DIMU_OF = extractIMUAccFromLinFile(lin, hub=2, ptfm=2, nac=0, vel=True)
            #CIMU_OF = CIMU_OF.rename(columns=ColMapLinFile, index=ColMapLinFile)
            #DIMU_OF = DIMU_OF.rename(columns=ColMapLinFile, index=ColMapLinFile)

            # --- FASTLinModelFTNSB is an instance of FASTLinModel, instance of LinearStateSpace
            # Used to handle one or several lin files
            linmodel = FASTLinModelFTNSB(fstFilename=fstLin, usePickle=usePickle)
            if linmodel.pickleFile is None:
                linmodel.save() # No pickle file was found (.lin was read, we save the pickle to speedup)
            linmodel.rename(verbose=False)
            #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> lin model')
            #print(linmodel)
            A_OF, B, C, D = linmodel.toDataFrames()
            #linmodel.extract(sX=sX, sU=sU, sY=sY, check=False)
            # Chose between OF or YAMS
            A=A_OF


        # --- 
        A[abs(A)<zero_threshold]=0
        B[abs(B)<zero_threshold]=0
        C[abs(C)<zero_threshold]=0
        D[abs(D)<zero_threshold]=0


        # --------------------------------------------------------------------------------}
        # ---  Kalman model (Augmented/modified physical model)
        # --------------------------------------------------------------------------------{
        KM.sStates    =      DOFs.split(',')
        KM.sStates   +=     ['d'+s for s in KM.sStates] # Add derivatives
        KM.sStatesD  =      ['d'+s for s in KM.sStates] # All states derivatives
        KM.sInp        = np.array([s.strip() for s in sInp.split(',') ])  if len(sInp)>0 else np.array([])
        KM.sMeas       = np.array([s.strip() for s in sMeas.split(',')]) if len(sMeas)>0 else np.array([])
        KM.sStor       = np.array([s.strip() for s in sStor.split(',')]) if len(sStor)>0 else np.array([])
        KM.sAug        = np.array([s.strip() for s in sAug.split(',') ])  if len(sAug)>0 else np.array([])
        KM.sStates     = np.array(KM.sStates)

        # --- Build linear system
        nX = len(KM.sStates)+len(KM.sAug)
        nU = len(KM.sInp   )
        nY = len(KM.sMeas  )
        nq = len(KM.sStates)
        sX0= list(KM.sStates)
        sX = list(KM.sStates)+list(KM.sAug)
        sqd = KM.sStatesD
        sU = KM.sInp
        sY = KM.sMeas

        Xx, Xu, Yx, Yu = EmptyStateMat(nX, nU, nY)
        Xx, Xu, Yx, Yu = EmptyStateDF(nX,nU,nY,sX,sU,sY)

        # --------------------------------------------------------------------------------}
        # --- Filling state matrix Xx
        # --------------------------------------------------------------------------------{

        # basic A matrix
        sXd = ['d'+s for s in sX]
        for sx in A.index:
            if sx not in Xx.index:
                raise Exception('{} not present in Xx ({})'.format(sx, Xx.index))
            for sx2 in A.columns:
                Xx.loc[sx,sx2] = A.loc[sx,sx2]

        # --- Hard Coding
        def setter(MM, sM, srow, scol, value, verbose=True):
            if srow not in MM.index:
                print('[WARN] Matrix {}: Row {} is not present in matrix {}'.format(sM, srow))
                return
            if scol not in MM.columns:
                print('[WARN] Matrix {}: Col {} is not present in matrix {}'.format(sM, srow))
                return
            if verbose:
                print('[INFO] Matrix {}: Replacing [{:10s} x  {:10s}] from {:} to {} '.format(sM, srow, scol, MM.loc[srow, scol], value))
            MM.loc[srow,scol]  = value

        # --- Who influences omega
        if not fullColumns:
            for s in KM.sStates:
                setter(Xx, 'Xx', 'dpsi' , s, 0) 
                setter(Xx, 'Xx', 'ddpsi', s, 0) 
        else:
            setter(Xx, 'Xx', 'ddpsi', 'psi', 0) 


        # --- Main B matrix
        # Thrust fay faz Qaero may maz
        colAugForce = mainLinInputs(hub=2, nac=1, ptfm=2, gen=1, pitch=1)
        colAugForce2 = renameList(colAugForce, ColMapLinFile)
        B = subMat(B, rows=None, cols=colAugForce2, check=True)

        # --- Rotor Inertia
        J_LSS_YAMS = linmodel.WT.rot.inertia[0,0] 
        J_LSS_OF_Qgen   = -1/B.loc['ddpsi','Qgen']
        J_LSS_OF_Qaero  =  1/B.loc['ddpsi','Qaero']
        J_LSS = J_LSS_OF_Qgen # Selection
        print('[INFO] Rotor Inertia seleted: {}'.format(J_LSS))

        if 'Qaero' in KM.sAug:
            if fullColumns:
                Xx.loc[sqd, 'Qaero'] = B.loc[sqd, 'Qaero'] # <<<<<
            setter(Xx, 'Xx', 'ddpsi', 'Qaero', 1/J_LSS)
        if 'Qgen' in KM.sAug:
            if fullColumns:
                Xx.loc[sqd, 'Qgen'] = B.loc[sqd, 'Qgen'] # <<<<<
            setter(Xx, 'Xx', 'ddpsi', 'Qgen' , -1/J_LSS)
        if 'Qgen' in KM.sInp:
            if fullColumns:
                Xu.loc[sqd, 'Qgen'] = B.loc[sqd, 'Qgen'] # <<<<<
            setter(Xu, 'Xu', 'ddpsi', 'Qgen' ,-1/J_LSS)

        # --- Thrust
        BFHx = B.loc[sqd, 'Thrust'] # Hub x force
        BFNx = B.loc[sqd, 'NacFxN1_[N]'] # Nacelle x force
        BFx_selected = B.loc[sqd, sThrust]*kThrustA
        print('[INFO] Thrust ddq relation: {}'.format(BFx_selected.loc['ddq_FA1']))
        if 'Thrust' in KM.sAug:
            if fullColumns:
                Xx.loc[sqd, 'Thrust'] = BFx_selected.loc[sqd]
            Xx.loc['ddq_FA1','Thrust'] = BFx_selected.loc['ddq_FA1']
        if 'Thrust' in KM.sInp:
            if fullColumns:
                Xu.loc[sqd, 'Thrust'] = BFx_selected.loc[sqd]
            Xu.loc['ddq_FA1','Thrust'] = BFx_selected.loc['ddq_FA1']

        # --------------------------------------------------------------------------------}
        # --- Filling output matrix Yx
        # --------------------------------------------------------------------------------{
        # States directly measured (States that are in Y directly)
        sYX = [sx for sx in sX if sx in sY] # States that are in Y
        for sxy in sYX:
            sy=sxy
            Yx.loc[sy,sxy] = 1
        # IMU
        colAugForce = mainLinInputs(hub=2, nac=1, ptfm=2, gen=1, pitch=1)
        colAugForce2 = renameList(colAugForce, ColMapLinFile)
        _,_,CIMU, DIMU = linmodel.extract(sU=colAugForce2, sY=sIMU2, verbose=False, check=False, inPlace=False)
#         else:
#             raise NotImplementedError()
        sYIMU = [sy for sy in sIMU2 if sy in sY]
        for sy in sYIMU:
            for sx in CIMU.columns:
                Yx.loc[sy,sx] = CIMU.loc[sy,sx]
            for sx in DIMU.columns:
                if sx in Yx.columns: # Augmented states
                    Yx.loc[sy,sx] = DIMU.loc[sy,sx]
        if 'Thrust' in Yx.columns:
            if fullColumns:
                Yx.loc[sYIMU,'Thrust'] = DIMU.loc[sYIMU,sThrust]*kThrust
            else:
                Yx.loc[sYIMU,'Thrust'] = 0
                Yx.loc['NcIMUAx','Thrust'] = DIMU.loc['NcIMUAx',sThrust]*kThrust


        # --------------------------------------------------------------------------------}
        # ---  Yu matrix
        # --------------------------------------------------------------------------------{
        # --- Inputs directly measured (Inputs that are in Y directly)
        sUY = [su for su in sU if su in sY] # Inputs that are in Y
        for su in sUY:
            Yu.loc[su,su] = 1
        # --- IMU
        for sy in sYIMU:
            for sx in DIMU.columns:
                if sx in Yu.columns:
                    Yu.loc[sy,sx] = DIMU.loc[sy,sx]

        if 'Thrust' in Yu.columns:
            if fullColumns:
                Yu.loc[sYIMU,'Thrust'] = DIMU.loc[sYIMU,sThrust]*kThrust
            else:
                Yu.loc[sYIMU,'Thrust'] = 0
                Yu.loc['NcIMUAx','Thrust'] = DIMU.loc['NcIMUAx',sThrust]*kThrust

        # --- HACK Attempts
        # Heave is a bit too crazy
        Yx.loc['NcIMUAz','z'] *= kIMUz_z

#         printMat('Xx',Xx, xmin=1e-8)
#         printMat('Yx',Yx, xmin=1e-8)
        print('Xx\n',Xx)
        print('Yx\n',Yx)
        print('Xu\n',Xu)
        print('Yu\n',Yu)

        KM.Xx = Xx
        KM.Xu = Xu
        KM.Yx = Yx
        KM.Yu = Yu

        import control
        O   = control.obsv(Xx,Yx)
        sys = control.StateSpace(Xx,Xu,Yx,Yu)
        try:
            Wc = control.gram(sys, 'c')
        except:
            print('[FAIL] gramian Controlability')
            pass
        try:
            print('[FAIL] gramian Observability')
            Wo  = control.gram(sys, 'o')
        except:
            pass

# --- 
KM = KalmanModel(modelName, fstSim, fstLin, DOFs=DOFs, sInp=sInp, sAug=sAug, sStor=sStor, sMeas=sMeas)

# --- Actual Kalman filter simulation
KF=KalmanFilterFTNSLin(KM, fstSim, WSE_pickleFile=WSE_pklFilename)
MeasFile   = fstSim.replace('.fst','.outb')

# --- Loading "Measurements" (Defining "clean" values, estimate sigmas from measurements)
ColMap     = KM.ColMap
KF.loadMeasurements(MeasFile, nUnderSamp=nUnderSamp, tRange=tRange, ColMap=ColMap)

# --- Tuning of Sigmans
# KF.sigY['z']/=10000
# KF.sigY['NcIMUAz']*=100
# KF.sigX['z']=1
# KF.sigY=sigY
# for k,v in KF.sigX.items():
#     KF.sigX[k]=v/10
# for k,v in KF.sigY.items():
#     KF.sigY[k]=v/10
KF.sigX['Qaero']*=kSigQaero
KF.print_sigmas()

# --- Storage for plot and setting up covariances from sigmas
KF.prepareTimeStepping() 

# --- Creating noise measuremnts
KF.prepareMeasurements(NoiseRFactor=NoiseRFactor, bFilterAcc=bFilterAcc, nFilt=nFilt)

# --- Set Initial conditions
x = KF.initFromClean()

KF.timeLoop()

fig = KF.plot_X(nPlotCols=2, figSize=(12.8,8.2), title='States - LinFile:{}'.format(labelLin))
# fig.subplots_adjust(left=0.12, right=0.98, top=0.955, bottom=0.12, hspace=0.20, wspace=0.20)
figName = fstSim.replace('.fst','_KF_{}.png'.format(labelLin))
print('>>> FIG', figName)
fig.savefig(figName)

# KF.plot_U() # TODO
KF.plot_Y()
KF.plot_S()
plt.show()

# 
# # --- Time loop
# if debug:
#     print(OutputFile)
# KF.timeLoop()
# KF.moments()
# 
# if bExport:
#     KF.export(OutputFile)
# return KF
# 

# import pdb; pdb.set_trace()
