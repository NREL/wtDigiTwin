[![Build Status](https://travis-ci.com/ebranlard/wtDigitwin.svg?branch=master)](https://travis-ci.com/ebranlard/wtDigiTwin)

# wtDigiTwin
A digital twin model for wind turbine. Contains a set of structrual dynamics tools, beam theory, FEM and more.


This wind turbine digital twin software (wtDigiTwin) provides a digital twin solution for wind turbine applications. The focus of wtDigiTwin is to estimate loads, motions and environmental conditions for an operating wind turbine. The program uses supervisory control and data acquisition (SCADA) measurements as inputs, together with a wind turbine model. 

The wind industry is currently challenged by the high cost of operation and maintenance. These costs could be mitigated if component failures are predicted, but such predictions are difficult unless the turbines are equipped with expensive measuring devices. The alternative is to use a digital twin such as wtDigiTwin to estimate the necessary signals. 

wtDigiTwin can perform online prediction of signals that are otherwise not measured, using a limited set of reliable measurements and a physics-based model. The predicted signals can be used in applications that have direct cost benefits: 1) real-time estimation of the fatigue consumption of key components of the wind turbine; 2) root cause analyses and failure detections ; 3) lifetime reassessments ; 4) improvements to follow-on designs. 

The current version provides examples to estimate wind speed, thrust, torque, tower-top position, and tower loads on an onshore wind turbine using the following measurements tower top acceleration, generator torque, pitch, and rotational speed. The model combines a linear state-space model, a wind speed estimator, and a Kalman filter algorithm that integrates measurements with the state model to perform state estimations. The state space model is obtained either using OpenFAST linearizations, or using the yams package provided with the software.





## QuickStart
NOTE: This repository relies on the `welib` submodule, remember to include `--recurse-submodules` when cloning it. 

Download, install dependencies and package:
```bash
git clone --recurse-submodules https://github.com/NREL/wtDigiTwin
cd wtDigiTwin
python -m pip install --user -r requirements.txt  
python -m pip install -e .      # install
python -m unittest discover -v  # run test
```


## Examples
Simple working examples are provided in the example directory.


## Relevant packages from welib
The repository relies heavily on [welib](http://github.com/ebranlard/welib/) 

In particular the following subpackages of welib:

- yams: multibody analyses
- beams: analytical results for beams
- fast: tools to handle OpenFAST models, in particular a linear OpenFAST model
- fem: Finite Element Method tools (beams)
- kalman: kalman filter
- system: tools for dydnamic systems (e.g. LTI, state space) and mechanical systems (M,C,K matrices), eigenvalue analysis, time integration
- ode: tools for time integration of ODE
- ws\_estimator: wind speed estimator for wind energy based on tabulated Cp Ct


Some of the packages have dependency with the [weio](http://github.com/ebranlard/weio/) library to read and write files.


## How to cite:

The core of the model for onshore application is given in the following article:   

- Branlard,E, Giardina, D., Brown, C. S. , *Augmented Kalman filter with a reduced mechanical model to estimate tower loads on a land-based wind turbine: a step towards digital-twin simulations*, 2020 [link](https://doi.org/10.5194/wes-5-1155-2020)


Applications using OpenFAST linearization were presented in the following work:

- Branlard,E, Jonkman, J., Dana, S., Doubrawa, P., *A digital twin based on OpenFAST linearizations for real-time load and fatigue estimation of land-based turbines*, 2020 [link](https://iopscience.iop.org/article/10.1088/1742-6596/1618/2/022030)


The structural model referred to as YAMS was described in the following:

 - Branlard,E, *Flexible multibody dynamics using joint coordinates and the Rayleigh‐Ritz approximation: The general framework behind and beyond Flex* , 2019, [link](https://onlinelibrary.wiley.com/doi/abs/10.1002/we.2327). A pre-print of this article is available in the `_doc` folder of this repository.

