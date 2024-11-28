""""
Example script demonstrating how to use the NLP solver to determine the optimal speed and charging profile of an electric vehicle (EV) in order to minimize the total trip time. 

The controller plans the car's speed trajectory such that the given route is traversed in the shortest time possible, while simultaneously  respecting speed limits as well as vehicle's and battery's technical requirements.

We consider a single electric vehicle and a fixed route with given road slopes (i.e. angles) and speed limits. Optionally, some charging stations could be available at predefined locations on the road. The user can define whether or not the vehicle has to stop at every charging point or 
the optimizer can decide internally. This is achieved using the `trip.reduceSpeed` option. However, the accuracy and feasibility of the solution are not guaranteed in the latter case, as the formulation does not explicitly cover discrete decisions.

Due to the spatial characteristic of the problem, the formulation is discretized in space domain. Vehicle dynamics are described discretely using kinetic energy laws and longitudinal velocity and force action. In terms of vehicle energetics, the SoC is regulated through charging and energy dissipation due to driving. 

The objective is to minimize time as terminal state as well as to reduce cost associated with slack variable used to relax the min/max SoC bounds.

The problem is currently solved as a full-horizon MPC snapshot.

Variables are collected stage-wise into 

    z = [slack Ft Fb deltaTch v t SoC].

See also FORCES_NLP.

NOTE: The user is expected to define necessary trip parameters directly in :class:`TripParameters`!

This file is part of the FORCESPRO client software for Python.
(c) embotech AG, 2013-2023, Zurich, Switzerland. All rights reserved.
"""

import numpy as np
import math
import casadi
import forcespro.nlp
import forcespro.modelling
import matplotlib
import matplotlib.pyplot as plt

class TripParameters:
    """Trip parameters (to be defined by the user)"""    
    def __init__(self):        
        # Note: Accuracy and feasibility of the solution are not guaranteed when the `trip.reduceSpeed` flag is disabled due to inherent mixed-integer nature of the problem!

        # Trip type (0 - long (Munich - Cologne); 1 - short)
        self.type = 1
        
        if self.type == 1:
            # Spatial discretization step (km)
            self.Ls = 1
            # Trip distance (km) 
            self.dist = 50
            # EV charging station locations (km)
            self.kmPosCh = np.array([8, 18, 30, 40, 45])
            # Force vehicle to reduce speed at charging locations
            self.reduceSpeed = False
            # Initial vehicle speed (km/h)
            self.initSpeed = 30
            # Initial SoC [0, 1]
            self.initSoC = 0.25
        else:
            # Spatial discretization step (km)
            self.Ls = 1
            # Trip distance (km) 
            self.dist = 573
            # EV charging station locations (km)
            self.kmPosCh = np.array([110, 150, 250, 375])
            # Force vehicle to reduce speed at charging locations
            self.reduceSpeed = True
            # Initial vehicle speed (km/h)
            self.initSpeed = 30
            # Initial SoC [0, 1]
            self.initSoC = 0.75

        self.checkTripSetup()
        
    def checkTripSetup(self):
        """Checks validity of provided trip setup and modifies it if needed"""
        # Trip type has to be 0 or 1
        assert self.type in [0, 1], 'Incorrect input: trip type should be 0 or 1.'        
        # Force vehicle to reduce speed at charging locations when running shrinking horizon MPC or the long trip scenario
        if self.type==0:
            self.reduceSpeed = True   

class VehicleParameters:        
    """Constant and variable vehicle parameters (not to be changed)"""    
    def __init__(self, trip):
        # Constant vehicle parameters
        # The data is obtained from "Technical specifications of the BMW i3 (120 Ah)" valid from 11/2018 and available at https://www.press.bmwgroup.com/global/article/detail/T0285608EN/
        # ---------------------------

        # Gravitational acceleration (m/s^2)
        self.g = 9.81
        # Vehicle mass (kg)
        self.mv = 1345
        # Mass factor (-)
        self.eI = 1.06
        # Equivalent mass (kg)
        self.meq = (1 + self.eI) * self.mv
        # Projected frontal area (m^2)
        self.Af = 2.38
        # Air density (kg/m^3)
        self.rhoa = 1.206
        # Air drag coefficient (-)
        self.ca = 0.29
        # Rolling resistance coefficient (-)
        self.cr = 0.01    
        # Maximum traction force (N)         
        self.FtMax = 5e3    
        # Minimum traction force (N) 
        self.FtMin = 0
        # self.FtMin = -1.11e3
        # Maximum breaking force (N)
        self.FbMax = 10e3
        # Battery's energy capacity (Wh) 
        # BMW i3 60Ah - 18.2kWh BMW i3 94Ah - 27.2kWh BMW i3 120Ah - 37.9kWh
        self.Ecap = 37.9e3
        # Maximum permissible charging time (s)
        self.TchMax = 3600
        # Charging power (W) 
        # 7.4 kW on-board charger on IEC Combo AC, optional 50 kW Combo DC
        self.PchMax = 50e3
        # Power dissipation factor (-) (adjusted for Munich - Cologne trip in order to achieve realistic BMW i3 range of ~260km)
        if trip.type == 1:
            self.pf = 1
        else:
            self.pf = 0.4
        # Minimum vehicle velocity (km/h)    
        self.vMin = 30
        # Maximum vehicle velocity (km/h)
        self.vMax = 150  
        # Minimum SoC (-)    
        self.SoCmin = 0.1
        # Maximum SoC (-)    
        self.SoCmax = 0.9

        # Variable vehicle parameters
        # ---------------------------

        # Motor efficiency (-)
        self.eta = self.setupMotorEfficiency()
        # Charging power (W)
        self.Pch = self.setupChargingPower()
        # Traction force hyperbola (upper limit) (N)
        self.FtMaxHyp = self.setupTractionForceHyperbola()   

    def setupMotorEfficiency(self):
        """
        Returns cubic spline representing motor efficiency as a function of traction force.

        The data is taken from "Jia, Y.; Jibrin, R.; Goerges, D.: Energy-Optimal Adaptive Cruise Control for Electric Vehicles Based on Linear and Nonlinear Model Predictive Control. In: IEEE Transactions on Vehicular Technology, vol. 69, no. 12, pp. 14173-14187, Dec. 2020."
        """  
        FtSample = np.linspace(0, self.FtMax, 11)
        etaSample = np.array([0.835, 0.865, 0.9, 0.85, 0.79, 0.75, 0.72, 0.72, 0.72, 0.72, 0.72])       
        etaMotorSpline = forcespro.modelling.InterpolationFit(FtSample, etaSample)

        return etaMotorSpline

    def setupChargingPower(self):
        """
        Returns shape-preserving piecewise cubic spline representation of charging power as a function of SoC.
        
        The data is taken from Fastned charging chart available at https://support.fastned.nl/hc/en-gb/articles/204784718-Charging-with-a-BMW-i3

        NOTE: Matlab example uses piecewise linear approximation which might lead to negligible result discrepancies between the two clients when directly compared.
        """  
        socSample = np.array([0.15, 0.85, 1.0])
        PchSample = np.array([0.88, 1, 0.2]) * self.PchMax
        PchSpline = forcespro.modelling.InterpolationFit(socSample, PchSample, 'pchip')
                
        return PchSpline

    def setupTractionForceHyperbola(self):
        """
        Returns cubic spline representing motor efficiency as a function of traction force.
        
        The data is generated based on the technical article provided at https://x-engineer.org/need-gears/
        """  
        vSample = np.array([0.25, 0.4, 0.6, 0.8, 1.0]) * self.kmh2ms(self.vMax)
        FtMaxHypSample = np.array([1.0, 0.67, 0.43, 0.32, 0.28]) * self.FtMax
        FtMaxHypSpline = forcespro.modelling.InterpolationFit(vSample, FtMaxHypSample)
        
        return FtMaxHypSpline 

    def kmh2ms(self, kmh):
        """Converts km/h into m/s"""
        return kmh / 3.6
        
def generateSpeedProfile(trip, param):
    """"
    Formulates the optimization problem and generates a solver for the optimal speed & charging profile by calling the FORCESPRO code generation
    Assume variable ordering zi = [u{i}, x{i}] for i=1...N
    zi = [slack{i}, Ft{i}, Fb{i}, deltaTch{i}, v{i}, t{i}, SoC{i}]
    pi = [vMax{i}, vMin{i}, alpha{i}, TchMax{i}, Ls{k}] #k - iteration No. 
    """
    # Model Definition
    # ----------------
    
    # Problem dimensions    
    nx = 3    
    nu = 4
    npar = 5
    nh = 6
    
    model = forcespro.nlp.SymbolicModel()
    model.N = round(trip.dist / trip.Ls); # horizon length
    assert model.N % 1 == 0, 'Incorrect input: trip distance should be an exact multiple of the spatial discretization step'
    model.nvar = nx + nu;               # number of variables
    model.neq  = nx;                    # number of equality constraints
    model.nh   = nh;                    # number of inequality constraints
    model.npar = npar;                    # number of runtime parameters

    # Objective function
    R = np.diag([1e-7, 1e-6])
    slackCostFactor = 1e6
    model.objective = (lambda z: slackCostFactor * z[0] + casadi.horzcat(z[1], z[2]) @ R @ casadi.vertcat(z[1], z[2]))    
    model.objectiveN = (lambda z: z[5])

    # Dynamics, i.e. equality constraints   
    model.eq = lambda z, p: casadi.vertcat( np.sqrt(forcespro.modelling.smooth_max(2 * p[4] / 
                                            param.meq * (z[1] - z[2] - param.cr * param.mv * param.g * np.cos(p[2]) - param.mv * param.g * np.sin(p[2]) - 0.5 * param.ca * param.Af * param.rhoa * z[4]**2) + z[4]**2, 0)),
                                            z[5] + p[4] / z[4] + z[3],
                                            z[6] - param.pf * p[4] * z[1] / (3600 * param.eta(z[1])* param.Ecap) + param.Pch(z[6]) * z[3] / (3600 * param.Ecap))

    model.E = np.concatenate([np.zeros((nx, nu)), np.eye(nx)], axis=1)

    # Inequality constraints
    # Upper/lower variable bounds lb <= z <= ub
    #                             inputs                            |                states
    #                   slack       Ft     Fb           Tch                    v           t      SoC
    model.lb = np.array([0.,  param.FtMin, 0.,          0.,           kmh2ms(param.vMin),  0.,      0.])
    model.ub = np.array([0.1, param.FtMax, param.FbMax, param.TchMax, kmh2ms(param.vMax),  np.inf,  1.])

    # Nonlinear inequalities hl <= h(z,p) <= hu
    model.ineq = lambda z,p: casadi.vertcat(z[1] - param.FtMaxHyp(z[4]),
                                            z[3] - p[3],
                                            z[6] - param.SoCmax - z[0],
                                            param.SoCmin - z[6] - z[0],
                                            2 * p[4] / param.meq * (z[1] - z[2] - param.cr * param.mv * param.g * np.cos(p[2]) - param.mv * param.g * np.sin(p[2]) - 0.5 * param.ca * param.Af * param.rhoa * z[4]**2) + z[4]**2 - kmh2ms(forcespro.modelling.smooth_max((1 - z[3]) * p[0], param.vMin + 1))**2,
                                            kmh2ms(p[1])**2 - (2 * p[4] / param.meq * (z[1] - z[2] - param.cr * param.mv * param.g * np.cos(p[2]) - param.mv * param.g * np.sin(p[2]) - 0.5 * param.ca * param.Af * param.rhoa * z[4]**2) + z[4]**2))

    # Upper/lower bounds for inequalities                 
    model.hu = np.array([0,       0,        0,       0,      0,       0])
    model.hl = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])

    # Initial and final conditions
    model.xinitidx = np.arange(nu, nu + nx);  

    # Generate FORCESPRO solver    
    # -------------------------

    # Define solver options
    codeoptions = forcespro.CodeOptions("FORCESNLPsolver")
    codeoptions.printlevel = 0 
    codeoptions.nlp.compact_code = 1    

    # Generate code
    solver = model.generate_solver(codeoptions)

    return model, solver

def runSimulation(trip, param, model, solver):
    """Defines initial and stage-dependent runtime parameters and solves the problem"""
    # Define spatially distributed (i.e. stage-dependent) parameters 
    # --------------------------------------------------------------

    # Road speed limits and slope angles
    vMaxRoad, vMaxRoadStrict, vMinRoad, roadSlope = setupRoadParameters(param.vMin, trip)
    sim = {"vMaxRoad": vMaxRoad, "vMaxRoadStrict": vMaxRoadStrict, "vMinRoad": vMinRoad, "roadSlope": roadSlope}

    # Maximum permissible charging time
    deltaTchMax = setupChargingTime(param.TchMax, trip);    

    # Initialize the problem
    # ----------------------
    
    x0 = np.zeros(model.N * model.nvar)
    problem = {"x0": x0}
    kMax = int(trip.dist / trip.Ls)
    sim["kMax"] = kMax
    nx = model.neq
    nu = model.nvar - model.neq
    npar = model.npar

    assert trip.initSpeed >= param.vMin or trip.initSpeed <= param.vMax, 'Initial vehicle speed is outside of the speed limits.'
    assert trip.initSoC >= param.SoCmin or trip.initSoC <= param.SoCmax, 'Initial vehicle state-of-charge is outside of the limits.'   
        
    # X(1) = [v(1); t(1); SoC(1)]
    problem["xinit"] = np.array([kmh2ms(trip.initSpeed), 0, trip.initSoC])
    
    # Solve the problem
    # -----------------
    
    # Set runtime parameters
    problem["all_parameters"] = np.zeros(npar * model.N)
    for i in range(model.N):
        problem["all_parameters"][i * npar : (i + 1) * npar] = np.array([vMaxRoad[i], vMinRoad[i], roadSlope((i + 1) * trip.Ls * 1e3), deltaTchMax[i], trip.Ls * 1e3])  

    # Call solver
    solverout, exitflag, info = solver.solve(problem)    
    assert exitflag == 1, 'Some problem in solver.'    

    # Extract state and control vector        
    sim["exitflag"] = exitflag
    sim["Z"] = unpackDict(solverout)                
    sim["solvetime"] = info.solvetime
    sim["iters"] = info.it  

    sim = displayResults(sim)

    return sim 

def setupRoadParameters(vMin, trip):
    """
    Returns vectors of upper and lower speed limits and road slope as a function of spatial position
    """
    # Define maximum speed limit
    if trip.type == 1:
        # Arbitrarily generated short trip
        vLim = np.array([50.0, 80.0, 100.0, 120.0, 80.0])
        L = np.array([0.1, 0.1, 0.2, 0.4, 0.2]) * trip.dist / trip.Ls
        upperSpeedLimitStrict = np.concatenate([vLim[0] * np.ones(int(L[0])), vLim[1] * np.ones(int(L[1])), vLim[2] * np.ones(int(L[2])), vLim[3] * np.ones(int(L[3])), vLim[4] * np.ones(int(L[4]))])
        upperSpeedLimit = np.concatenate([vLim[0] * np.ones(int(L[0])), vLim[1] * np.ones(int(L[1])), vLim[2] * np.ones(int(L[2])), vLim[3] * np.ones(int(L[3])), vLim[4] * np.ones(int(L[4]))])
    else:
        # Munich - Cologne trip with simplified speed limits
        vLim = np.array([50, 100, 130, 150, 130, 100, 130, 150, 130, 150, 130, 100, 130, 100, 130, 100])
        L = np.array([0, 5, 15, 45, 105, 145, 155, 215, 260, 305, 375, 400, 410, 480, 490, 540, 550])
        upperSpeedLimitStrict = np.zeros(int(L[-1] / trip.Ls))
        upperSpeedLimit = np.zeros(int(L[-1] / trip.Ls))
        for i in range(len(vLim)):        
            upperSpeedLimitStrict[np.arange(L[i] / trip.Ls, L[i+1] / trip.Ls).astype(int)] = vLim[i] 
            upperSpeedLimit[np.arange(L[i] / trip.Ls, L[i+1] / trip.Ls).astype(int)] = vLim[i]

    # Define minimum speed limit
    lowerSpeedLimit = vMin * np.ones(len(upperSpeedLimitStrict))    

    # Define road slope
    if trip.type == 1:        
        alphaSample = np.array([0.05, 0.08, 0.14, 0.14, 0.2, 0.1, 0.025, 0.015, 0.05, -0.02, -0.08, -0.035, -0.015, 0.0, 0.05, 0.025, 0, 0.05, 0.08, 0.1, 0.15])        
    else:             
        alphaSample = np.array([0.04, 0.03, 0.05, 0.02, 0.04, 0.02, 0.03, 0.04, 0.03, 0.05, 0.03])
    sSample = np.linspace(0, trip.dist * 1e3, len(alphaSample))
    roadSlope = forcespro.modelling.InterpolationFit(sSample, alphaSample)
    
    # (Optional) Reduce maximum speed limit at charging locations
    if trip.reduceSpeed:
        chInd = trip.kmPosCh / trip.Ls
        # Not extending by default but could be changed by the user
        chIndExt = extendIndex(chInd, 0)
        upperSpeedLimit[chIndExt] = vMin + 0.1

    return upperSpeedLimit, upperSpeedLimitStrict, lowerSpeedLimit, roadSlope

def setupChargingTime(TchMax, trip):
    """Returns maximum permissible charging time as a function of spatial position"""
    chInd = trip.kmPosCh / trip.Ls
    # Not extending by default but could be changed by the user
    chIndExt = extendIndex(chInd, 0)
    deltaTchMax = 1e-2 * np.ones(int(trip.dist / trip.Ls));    
    deltaTchMax[chIndExt] = TchMax;  

    return deltaTchMax

def extendIndex(ind, n):
    """Extends index vector by adding n consecutive elements before and after each vector entry"""
    indExtended = np.zeros((2*n + 1) * len(ind), dtype=int)
    for i in range(len(ind)):
        indExtended[np.arange(i * (2*n + 1), (i + 1) * (2*n + 1))] = np.arange(ind[i] - n, ind[i] + n + 1)

    return indExtended

def displayResults(sim):
    """Computes and displays relevant simulation metrics"""
    triptime = sim["Z"][-1, 5]
    # Only consider meaningful charging stops (charging time > 5sec)
    chInd = sim["Z"][:, 3] > 5        
    chargetime = sum(sim["Z"][chInd, 3])
    numstops = sum(compactChargingInstances(chInd))
    
    sim["triptime"] = triptime 
    sim["numstops"] = numstops
    sim["chargetime"] = chargetime

    print(f"Total trip time: {round(triptime / 60, 2)} min")
    print(f"Total charging time: {round(chargetime / 60, 2)} min")
    print(f"Number of charging stops: {numstops}")

    return sim

def compactChargingInstances(chInd):
    """ Combine charging occurrence at two consecutive steps into one, since it is a consequence of a rounding error during charging location discretization"""    
    for i in range(len(chInd))[1: ]:
        if chInd[i] == 1 and chInd[i-1] == 1:
            chInd[i] = 0

    return chInd        

def plotResults(trip, param, sim):
    """Plot simulation results"""
    ind = np.arange(sim["kMax"])
    xkm = (ind + 1) * trip.Ls
    xm = np.arange(0, trip.dist * 1e3 + 100, 100)
    
    # Road speed limits and slope angles    
    vMaxRoadStrict = sim["vMaxRoadStrict"]
    vMinRoad = sim["vMinRoad"]
    roadSlope = sim["roadSlope"]
    roadSlopeSampled = np.zeros(len(xm))
    for i in range(len(xm)):
        roadSlopeSampled[i] = rad2deg(roadSlope(xm[i]))
      
    plt.style.use('seaborn')
    fig = plt.gcf()    
    fig.set_size_inches(10, 7)

    # Speed profile
    plt.subplot(4, 1, 1)
    plt.grid('both')         
    plt.step(xkm, vMaxRoadStrict[ind], '-.', color='tab:orange')
    plt.step(xkm, vMinRoad[ind], '-.', color='tab:orange')
    plt.step(xkm, ms2kmh(sim["Z"][:, 4]))   
    plt.xlim([trip.Ls, trip.dist] )
    plt.ylim([0, param.vMax])
    plt.xlabel('Distance [km]')
    plt.ylabel('Vehicle speed [km/h]')
    plt.title('Speed profile')    
    plt.gca().legend(["Speed limits"], loc="lower right")

    # Vehicle forces
    plt.subplot(4, 1, 2)
    plt.grid('both')    
    plt.step(xkm, sim["Z"][:, 1] * 1e-3)
    plt.step(xkm, sim["Z"][:, 2] * 1e-3)
    plt.step(xkm, np.minimum(param.FtMax * np.ones(len(ind)), param.FtMaxHyp(sim["Z"][:, 4])) * 1e-3, '-.', color='tab:orange', linewidth=1.0)
    plt.xlim([trip.Ls, trip.dist] )
    plt.ylim([param.FtMin * 1e-3, param.FbMax * 1e-3])
    plt.xlabel('Distance [km]')
    plt.ylabel('Force [kN]')
    plt.title('Vehicle forces')
    plt.gca().legend(["Traction force", "Breaking force", "Traction force limits"], loc="upper right")

    # Battery profile
    plt.subplot(4, 1, 3)
    plt.grid('both')    
    plt.step(xkm, 100 * param.SoCmin * np.ones(len(ind)), '-.', color='tab:orange')
    plt.step(xkm, 100 * param.SoCmax * np.ones(len(ind)), '-.', color='tab:orange')    
    plt.step(xkm, 100 * sim["Z"][:, 6])
    plt.xlim([trip.Ls, trip.dist] )
    plt.ylim([0, 100])
    plt.xlabel('Distance [km]')
    plt.ylabel('State of charge [%]', color='tab:blue')
    plt.title('Battery profile')
    plt.gca().legend(["SoC limits"], loc="center right")
    ax2 = plt.gca().twinx()    
    ax2.step(xkm, sim["Z"][:, 3] / 60, color='tab:green')
    plt.ylim([0, max(40, max(sim["Z"][:, 3] / 60))])
    ax2.set_ylabel('Charging time [min]', color='tab:green')
    plt.grid(False)

    # Road profile
    plt.subplot(4, 1, 4)
    plt.grid('both')        
    plt.step(xkm, vMaxRoadStrict[ind], color='tab:blue')
    plt.step(xkm, vMinRoad[ind], color='tab:blue')
    plt.xlim([trip.Ls, trip.dist])
    plt.ylim([0, param.vMax])
    plt.xlabel('Distance [km]')
    plt.ylabel('Speed limits [km/h]', color='tab:blue')
    plt.title('Road profile')    
    ax2 = plt.gca().twinx()    
    ax2.step(xm*1e-3, roadSlopeSampled, color='tab:green')
    ax2.step(xm*1e-3, np.zeros(len(xm)), '-.', color='tab:green')    
    plt.ylim([-20, 20])
    ax2.set_ylabel('Road slope [deg]', color='tab:green')
    plt.grid(False)
    
    fig.tight_layout()
    plt.show()
    
def unpackDict(dict):
    """Unpacks dictionary of 1D arrays into a 2D array by v-stacking them"""
    for key in dict.keys():
        row = dict[key]
        if key == list(dict.keys())[0]:
            dictUnpacked = row
        else:
            dictUnpacked = np.vstack((dictUnpacked, row))

    return dictUnpacked

def kmh2ms(kmh):
    """Converts km/h into m/s"""
    return kmh / 3.6

def ms2kmh(ms):
    """Converts m/s into km/h"""
    return ms * 3.6

def rad2deg(rad): 
    """Converts radians into degrees"""
    return rad*180/math.pi

def main():    
    """Main executable method"""
    # Initialize model (i.e. trip and vehicle) parameters
    trip = TripParameters()
    param = VehicleParameters(trip)

    # Formulate the problem and generate the solver
    model, solver = generateSpeedProfile(trip, param)

    # Simulate the model
    sim = runSimulation(trip, param, model, solver)

    # Plot simulation results
    plotResults(trip, param, sim)    

if __name__ == "__main__":
    main()    
    
