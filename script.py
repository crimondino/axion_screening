#!/usr/bin/env python3.7.16

# To run this script, in a separate terminal type:
#### conda activate conda_env
#### python3 script.py >> ./data/output.txt

import os,sys
print([ii for ii in sys.path])
sys.path.remove('/home/dpirvu/DarkPhotonxunWISE/hmvec-master')
sys.path.append('/home/dpirvu/axion/hmvec-master/')
sys.path.append('/home/dpirvu/python_stuff/')
print([ii for ii in sys.path])
import hmvec as hm
import numpy as np

from compute_power_spectra import *
from params import *
from plotting import *
import random

import functools
from concurrent.futures import ProcessPoolExecutor

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# Select electron profile
conv_gas  = True
conv_NFW  = False
pick_prof = (True if conv_gas else False)

# Select DP mass
maind=0

unWISEcol = 'blue'
pathdndz = "/home/dpirvu/DarkPhotonxunWISE/dataHOD/normalised_dndz_cosmos_0.txt"

# If parallelized
num_workers = 8

zthr  = 1.9
zreio = 1.9

MA = dictKey[maind]
zMin, zMax, rMin, rMax = chooseModel(MA, modelParams)

if conv_gas:
    name   = 'battagliaAGN'
    rscale = False
elif conv_NFW:
    rscale = True


# B-field stuff from Cristina
# See Table 1 of 2309.13104 for the halo properties for the given mass bins
file_names = ['profile_bfld_halo_1e10_h12.txt', 'profile_bfld_halo_1e10_h11.txt', 'profile_bfld_halo_1e11_h10.txt', 
              'profile_bfld_halo_1e11_h4.txt', 'profile_bfld_halo_1e12_h12.txt', 'profile_bfld_halo_1e13_h4.txt', 
              'profile_bfld_halo_1e13_h8.txt']
# file_names = ['profile_bfld_halo_'+str(i+1)+'.txt' for i in range(7)]# os.listdir('./data/bfield_profiles/')
mass_bins = 10.**np.array([9.9, 10.4, 10.9, 11.4, 12, 12.5, 13])

# Radial bins are the same for all of the files
rad_bins   = np.genfromtxt('./data/profiles/'+file_names[0], skip_header=3, max_rows=1)
rad_bins_c = rad_bins[:-1]+(rad_bins[1:]-rad_bins[:-1])/2.

Bfiled_grid = np.zeros((len(mass_bins), 66, 23))
logB_interp_list = []

for i, file in enumerate(file_names):
    # in gauss
    Bfiled_grid[i] = np.genfromtxt('./data/profiles/'+file_names[i], skip_header=7).astype(float)

  #  logB_interp_list.append(RegularGridInterpolator((np.log10(Bfiled_grid[i][::, 0]), rad_bins_c), \
  #                                                   np.log10(Bfiled_grid[i][::, 3:]), \
  #                                                   bounds_error=False, fill_value=-10))
    logB_interp_list.append(RegularGridInterpolator((np.log10(np.concatenate( (Bfiled_grid[i][::8, 0], Bfiled_grid[i][-1:, 0]) )), rad_bins_c),
                                                     np.log10(np.concatenate( (Bfiled_grid[i][::8, 3:], Bfiled_grid[i][-1:, 3:]) )),
                                                     bounds_error=False, fill_value=-10 ))

####### HALO MODEL ########

ellMax    = 9600
ells      = np.arange(ellMax)
chunksize = max(1, len(ells)//num_workers)

#hlil = 0.6766
#mMin = 7e8/hlil
mMin = 1e11
mMax = 1e17

zMax = min(zthr, zMax)
nZs  = 100
nMs  = 100
ms  = np.geomspace(mMin,mMax, nMs)               # masses
zs  = np.linspace(zMin, zMax, nZs)               # redshifts
rs  = np.linspace(rMin, rMax, 100000)            # halo radius
ks  = np.geomspace(1e-4,1e3, 5001)               # wavenumbers

print('Axion mass:', MA)
print('Halo masses:', mMin, mMax, nMs)
print('Redshifts:', zMin, zMax, nZs)

# Halo Model
hcos = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', mdef='vir', concmode='BHATTACHARYA', unwise_color=unWISEcol)
print('Test hmvec')
print(hm.default_params['H0'])
print(hcos.conc)

chis     = hcos.comoving_radial_distance(zs)
rvirs    = hcos.rvir(ms[None,:],zs[:,None])
cs       = hcos.concentration()
Hz       = hcos.h_of_z(zs)
nzm      = hcos.get_nzm()
biases   = hcos.get_bh()
deltav   = hcos.deltav(zs)
rhocritz = hcos.rho_critical_z(zs)
m200c, r200c = get_200critz(zs, ms, cs, rhocritz, deltav)

hod_name = "unWISE"+unWISEcol
hcos.add_hod(name=hod_name)

path_params = np.asarray([MA, nZs, zMin, zMax, ellMax, rscale])
path_params_thom = np.asarray([nZs, zMin, zreio, ellMax, rscale])

print('Importing CMB power spectra and adding temperature monopole.')
CMB_ps        = hcos.CMB_power_spectra()
unlenCMB      = CMB_ps['unlensed_scalar']
unlenCMB      = unlenCMB[:ellMax, :]
unlenCMB[0,0] = TCMB**2.
#lensedCMB     = CMB_ps['lensed_scalar']
#lensedCMB     = lensedCMB[:ellMax, :]
#lensedCMB[0,0]= TCMB**2.

dvols  = get_volume_conv(chis, Hz)

dothis = False
if dothis:
    rcross = get_rcross_per_halo(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, pick_prof, name=name)
    ucosth, angs = get_halo_skyprofile(zs, chis, rcross)
    
    prob = conv_prob(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=rscale, name=name)
    #prob = conv_prob_flat(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=rscale, name=name)

    print('Computing multipole expansion of angular probability u.')
    partial_u = functools.partial(get_uell0, angs, ucosth)
    with ProcessPoolExecutor(num_workers) as executor:
        uell0 = list(executor.map(partial_u, ells, chunksize=chunksize))

    prob00 = prob * uell0[0]
    avtau, dtaudz = get_avtau(zs, ms, nzm, dvols, prob00)

    np.save(data_path(*path_params), [rcross, prob, avtau, dtaudz, uell0])

    rcross, prob, avtau, dtaudz, uell0 = np.load(data_path(*path_params))
    probell = prob[None,...] * uell0

    print('Computing 1-halo angular PS.')
    Cl1h = get_Celldtaudtau_1h(zs, ms, ks, nzm, dvols, probell, ellMax)

    PzkLin     = hcos._get_matter_power(zs, ks, nonlinear=False)
    PzkLinz1z2 = (PzkLin[:,None,:] * PzkLin[None,...])**0.5
    dtaudz_ell = get_dtauell(ms, nzm, dvols, biases, probell)

    print('Computing 2-halo angular PS.')
    partial_get_2h = functools.partial(get_Celldtaudtau_2h, zs, ms, ks, chis, PzkLinz1z2, dtaudz_ell)
    with ProcessPoolExecutor(num_workers) as executor:
        Cl2h = list(executor.map(partial_get_2h, ells, chunksize=chunksize))

    np.save(cl_data_tautau_path(*path_params), [Cl1h, Cl2h])

    Cl1h, Cl2h = np.load(cl_data_tautau_path(*path_params))
    Cltot = Cl1h + Cl2h


print('Importing dark screening in temperature data.')
rcross, prob, avtau, dtaudz, uell0 = np.load(data_path(*path_params))
probell = prob[None,...] * uell0

Cl1h, Cl2h = np.load(cl_data_tautau_path(*path_params))
Cltot = Cl1h + Cl2h

llist, scrTT0 = get_Tmonopole_screeningPS(ellMax, ellMax, CMB=unlenCMB, DPCl=Cltot)
np.save(ClTT0_path(*path_params), [llist, scrTT0])

for ellMax in [1000, 2000, 5000, 9500]:
    dopowc = False
    dobisp = True
    dobslong = False

    interpkind = ('cubic' if maind>=2 else 'linear')

    l0Max, l1Max, l2Max = ellMax, ellMax, ellMax
    ells = np.arange(ellMax)
    print('Compute screening of polarization from axion. ellMAX = ', ellMax)

    path_params = np.asarray([MA, nZs, zMin, zMax, ellMax, rscale])

    # To compute the polarization screening, I am using the temperature screening monomopole
    # so neet to multiply by 3 because temperature only selected 1/3 polarization states, and also divide by 2, because now the in+out 
    # throgh halo crossing radius happen in uncollerated magnetic domains; 
    # the crossing still happens twice per halo, but now the in/out between themselves are uncorrelated.
    # therefore, for the two-point functions overall I need to multiply by 3**2. and divide by 2.
    probell_temp = np.sqrt(9./2.) * probell[:ellMax,...]

 #   for Npc in [1., 10., 100., 1000.]:
    for Npc in [1., 10.]:
        print('Doing', Npc, 'kiloparsecs per magnetic domain with ellMax=', ellMax)
        ellshort = np.array(np.arange(0, 10, 1).tolist() + 
                            np.arange(10, 100, 10).tolist() + 
                            np.arange(100, 1000, 75).tolist() + 
                            np.arange(1000, ellMax, 200).tolist() + 
                            [ellMax-1])
        ellshort = np.array(list(dict.fromkeys(ellshort)))
        ellshort = ellshort[ellshort<ellMax]
        print(np.shape(ellshort), ellshort)

        # this is only a function of redshift: l1, z1
        sigma, axEEell, axBBell = get_gaussian_pol_1kp(zs, chis, ellMax, Npc)
        Clpol = (axEEell + axBBell)
        print('Clpol', np.shape(Clpol))

        if dopowc:
            # this is the one-halo term of axion screening function of new l2 and redshift: l2, z1
            dCl1hdz = get_dCelldtaudtaudz_1h(zs, ms, ks, nzm, dvols, probell_temp, ellMax)
            print('dCl1hdz', np.shape(dCl1hdz))

            # Here we compute the integrand for the axion polarization 2-point function
            Clmix = np.trapz(dCl1hdz[ellshort,None,:] * Clpol[None,ellshort,:], zs, axis=-1)
            np.save(fullscr_polaxion_clmix_path(*path_params, Npc), Clmix)
            print('Clmix', np.shape(Clmix))

            # now I have to interpolate over the entire multipole range to do the sums weighted by wigner 3j symbols
            f = interp2d(ellshort, ellshort, Clmix, kind=interpkind)
            Clpolfull = f(ells, ells).T
            # Now we compute the sum over l1, l2 giving an array of the auto-correlation of axion polarization screening indexed by l.
            l0List, scrEE, scrBB = get_scrCLs_pol_axion(l0Max, l1Max, l2Max, Clpolfull, TCMB)
            np.save(fullscr_polaxion_tautau_path(*path_params, Npc), [l0List, scrEE, scrBB])
            print('Saved 1.')

        if dobisp:
            # This is the galaxy template defined by whether there is a central galaxy at a point or not
            # a function of z2 and m2
            Ncs  = hcos.hods[hod_name]['Nc']
            ngal = hcos.hods[hod_name]['ngal']
            dndz, zs, N_gtot, W_g, zsHOD, dndzHOD = get_dndz(pathdndz,dvols,zs)
            Nczm = W_g[:,None] * Ncs[None,:] / ngal[:,None]

            # Linear matter power function of z1, z2.
            PzkLin     = hcos._get_matter_power(zs, ks, nonlinear=False)
            PzkLinz1z2 = (PzkLin[:,None,:] * PzkLin[None,...])**0.5
            # Here we get the power spectrum corresponding to the halo-halo correlation function.
            # It is weighted by different functions that go into the expression for the bispectrum.
            # It is ultimately a function of l3, z1, m1, where z2 and m2 have been integrated out
            partial_get_clin = functools.partial(get_dCelllin, zs, ms, ks, chis, nzm, dvols, biases, Nczm, PzkLinz1z2)
            with ProcessPoolExecutor(num_workers) as executor:
                dCellhhdzdm = np.array(list(executor.map(partial_get_clin, ellshort, chunksize=chunksize)))
            np.save(fullscr_polaxion_clhh_path(*path_params, Npc), dCellhhdzdm)
            print('Saved Cell halo-halo dCellhhdzdm:', np.shape(dCellhhdzdm))

            # Now we compute the C_{\ell}^{\tau\tau,1-halo}, which is the same factor we had inside the axion polarization screening power spectrum
            # but without having integrated the mass and redshift; this has shape: l2, z1, m1,
            # Here l2 is not the ell in the reduced bispectrum, but the l2 that comes from combining angular momenta together with the magnetic field domain shape
            dCl1hdzdm = get_dCelldtaudtaudzdm_1h(zs, ms, ks, nzm, dvols, probell_temp, ellMax)
            print('dCl1hdzdm', np.shape(dCl1hdzdm))

            # Here the C_{\ell}^{\tau\tau,1-halo} is multiplied by C_{\ell}^{halo-halo} and now we can integrate out the m1 list
            # this now has shape: l3, l2, z1
            Clmix = np.trapz(dCl1hdzdm[None,ellshort,:,:] * dCellhhdzdm[:,None,:,:], ms, axis=-1)
            print('Clmix', np.shape(Clmix))

            # Here we finalize the calculation of the bispectrum term which must be summed over by multiplying by the magnetic field domain shape
            # Integrating out z1, this now has shape l3, l2, l1 where l2 and l1 will be summed over, weighted by the appropriate wigner 3j symbols
            Blmixx = np.trapz(Clmix[:,:,None,:] * Clpol[None,None,ellshort,:], zs, axis=-1)
            np.save(fullscr_polaxion_bisp_integrand_path(*path_params, Npc), Blmixx)
            print('Bispectrum integrand Blmixx done', np.shape(Blmixx))

            if dobslong:
                print('doing bispectrum for ellMax, Npc = ', ellMax, Npc)

                #Keeping l3 fixed, we want to sum over l1,l2 weighting the appropriate wigner 3j symbols
                print('Starting 2.')
                Bl3llinterpolated = np.zeros((len(ellshort), ellMax, ellMax))
                for li3 in range(len(ellshort)):
                    f = interp2d(ellshort, ellshort, Blmixx[li3,:,:], kind=interpkind)
                    Bl3llinterpolated[li3, :, :] = f(ells, ells).T

                Bl3llreducedEEg, Bl3llreducedBBg = get_scrBLs_polaxion_gal(l0Max, l1Max, l2Max, Bl3llinterpolated)

                np.save(fullscr_reducedbisp_tautau_path(*path_params, Npc), [Bl3llreducedEEg, Bl3llreducedBBg])
                print('Saved 2.')
            else:
                print('doing approximate bispectrum for ellMax, Npc = ', ellMax, Npc)

                #For the constant bispectrum integrand approximation, which holds when npc < 10, we need only compute
                # l1=l2=L at a single value l=0 for EEg and l1=l2=L at a single value l=1 for BBg
                # So once we interpolate the Blmix, we can take the diagonal terms only in the 2nd and 3rd dimension:
                print('Starting 2.')
                Bl3llinterpolated = np.zeros((len(ellshort), ellMax, ellMax))
                for li3 in range(len(ellshort)):
                    f = interp2d(ellshort, ellshort, Blmixx[li3,:,:], kind=interpkind)
                    Bl3llinterpolated[li3, :, :] = f(ells, ells).T

                Bl0reducedEEg, Bl1reducedBBg = get_scrBLs_approximation_polaxion_gal(l1Max, Bl3llinterpolated)

                Bl3llreducedEEg, Bl3llreducedBBg = np.zeros((2, len(ellshort), ellMax))
                for li in range(len(ellshort)):
                    Bl3llreducedEEg[li,2:] = Bl0reducedEEg[li]
                    Bl3llreducedBBg[li,2:] = Bl1reducedBBg[li]
                np.save(fullscr_approx_reducedbisp_tautau_path(*path_params, Npc), [Bl3llreducedEEg, Bl3llreducedBBg])

                

print('All Done.')
