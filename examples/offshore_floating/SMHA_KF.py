""" 
Perform a Digital twin simulation of the NREL 5-MW Spar
The Wind Speed Estimator is not used in this script.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from wtDigiTwin.FTNS_DigitalTwin import DigitalTwin
scriptDir=os.path.dirname(__file__)
np.set_printoptions(linewidth=300, precision=2)

# --------------------------------------------------------------------------------
# --- Parameters for digital twin simulation
# --------------------------------------------------------------------------------
# --- Turbine model
nGear=97 # gear ratio
fstLin = os.path.join(scriptDir,'../../_data/Spar_lin/Main.fst');  

# --- Wind speed estimator (WSE) 
WSE_pklFilename = None # TODO provide a pickle file to be able to estimate thrust and wind speed

# --- Measurement 
fstSim = os.path.join(scriptDir,'../../_data/Spar_sim/Main.fst'); 
MeasFile = fstSim.replace('.fst','.outb') # Here we use an OpenFAST simulation as "measurements"
nUnderSamp   = 0     # Undersampling of measurement inputs
NoiseRFactor = 0.0   # Signal to noise ratio to add to measurements
bFilterAcc   = False # Filter the accelration measurements
nFilt        = 15    # 

# --- State estimator settings
# State-space model (Q: states, Qa: augmented states, Y: outputs, U: inputs, S: storage)
sQ  = 'x,y,z,phi_x,phi_y,phi_z,q_FA1,psi,dx,dy,dz,dphi_x,dphi_y,dphi_z,dq_FA1,dpsi'
sQa = 'Qaero'
sY  = 'x,y,phi_x,phi_y,dpsi,NcIMUAx,NcIMUAy,NcIMUAz,Qgen'
sU  = 'Qgen,pitch,Thrust'
sS  = '' # e.g. WS
#sS += ','.join(['TwHt{}MLyt_[kN-m]'.format(i+1) for i in Isec])
# Tuning option for state estimator
tuning={} 
tuning['zero_threshold'] = 1e-9          # Watch out for 1/J ~ e-8
tuning['sThrust']        = 'NacFxN1_[N]'
tuning['sThrust']        = 'Thrust'
tuning['kThrust']        = 0.3           # Tuning factor for thrust measured acceleration feedback
tuning['kThrustA']       = 1             # Tuning factor for thrust state accelerations
tuning['kIMUz_z']        = 3             # Tuning factor z into IMUz
tuning['kSigQaero']      = 1             # Tuning of covariance for aero torque
tuning['fullColumns']    = True          # Use full columns of lin matrices or key values of it
# Combining state Estimator Options into one dictionary
SEOpts={'fstLin':fstLin, 'sQ':sQ, 'sU':sU, 'sQa':sQa, 'sS':sS, 'sY':sY, 'nGear':nGear, 'tuning':tuning}

# --------------------------------------------------------------------------------
# --- Digital twin simulation
# --------------------------------------------------------------------------------
# --- Initialization of the digital twin with aero estimator, state estimator and virtual sensing
DT = DigitalTwin()
DT.setupAeroEstimator(fstSim, WSE_pklFilename)
DT.setupStateEstimator(**SEOpts)
DT.setupVirtualSensing(vsType='SL_YAMS', fstFile=fstSim)
# --- Load measurement data (given time series)
DT.setupMeasurementData(MeasFile, tRange=None, nUnderSamp=nUnderSamp, bFilterAcc=bFilterAcc, nFilt=nFilt, NoiseRFactor=NoiseRFactor, tuning=tuning)
# --- Perform digital twin simulation for that measurement timeseries
KF = DT.timeLoop(virtualSensing=False)

# --- Export results to file
dfAll = KF.toDataFrame()
resOut = fstSim.replace('.fst','_KF.outb')
dfAll = KF.saveOutputs(resOut)

# --- Simple plots
figX = KF.plot_X()
figY = KF.plot_Y()
figS = KF.plot_S()
figU = KF.plot_U()

plt.show()

