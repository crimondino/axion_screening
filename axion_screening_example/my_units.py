import math
import numpy as np

# Notebook for user defined units. All dimensional quantities are in GeV = 1
GeV = 1; 
TeV = 10**3 * GeV;
MeV = 10**-3 * GeV; keV = 10**-3 * MeV; eV = 10**-3 * keV; meV = 10**-3 * eV; 

Kg = 5.6096 * 10**26 * GeV
Gram = 10**-3 * Kg
Meter = 1/(0.1973 * 10**-15 * GeV)
CentiMeter = 10**-2 * Meter
FemtoMeter = 10**-15 * Meter;  KiloMeter = 10**3 * Meter;
Second = 2.99792458 * 10**8 * Meter
Hz = Second**-1
kHz = 10**3 * Hz; MHz = 10**6 * Hz; GHz = 10**9 * Hz; THz = 10**12 * Hz; mHz = 10**-3 * Hz;
Hour = 3600*Second
Year = 365*24*Hour

Kelvin = 8.6 * 10**-5 * eV
Joule = Kg * Meter**2 * Second**-2; erg = 10**-7 * Joule; Watt = Joule * Second**-1;
Newton = Kg * Meter * Second**-2; Pa = Kg * Meter**-1 * Second**-2; GPa = 10**9 * Pa;

MPlanck = 1.2209*math.pow(10, 19)
GN = math.pow(MPlanck, -2)
mPlanck = MPlanck/math.sqrt(8*math.pi)

kpc = 3261*Year;
Mpc = math.pow(10, 3)*kpc
pc = math.pow(10, -3)*kpc
AU = 150 * 10**6 * 10**3 * Meter;
RSolar = 695.51 * 10**6 * Meter; MSolar = 1.98 * 10**30 * Kg; 

Hubble0 = 67.8*math.pow(10, 3)*Meter/Second/Mpc
zeq = 3250
HubbleEq = Hubble0*math.pow(1+zeq, 3/2) 
aeq = 1/(1 + zeq)
RhoCrit = (3*math.pow(Hubble0,2))/(8*math.pi*GN)
hubble0 = Hubble0/(100*10000*Meter/Second/Mpc)

Tesla = 195 * eV**2; Gauss = 10**-4 * Tesla;
AlphaEM = 1/137;
ElectronCharge  = np.sqrt(4 * np.pi * AlphaEM); Coulomb = (5.28 * 10**-19)**-1;

MProton = 0.938 * GeV; 
MElectron = 511. * keV;