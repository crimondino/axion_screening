#!/usr/bin/env python3.7.16

# To run this script, in a separate terminal type:
#### conda activate conda_env
#### python3 script.py >> ./data/output.txt

import os,sys
print([ii for ii in sys.path])
#sys.path.remove('/home/dpirvu/DarkPhotonxunWISE/hmvec-master')
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
num_workers = 40

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

ellMax0   = 9600
ells      = np.arange(ellMax0)
chunksize = max(1, len(ells)//num_workers)

#hlil = 0.6766
#mMin = 7e8/hlil
mMin = 1e11
mMax = 1e17

zMax = min(zthr, zMax)
nZs  = 50
nMs  = 100
ms  = np.geomspace(mMin,mMax, nMs)               # masses
zs  = np.linspace(zMin, zMax, nZs)               # redshifts
rs  = np.linspace(rMin, rMax, 100000)            # halo radius
ks  = np.geomspace(1e-4,1e3, 5001)               # wavenumbers

print('Axion mass:', MA)
print('Halo masses:', mMin, mMax, nMs)
print('Redshifts:', zMin, zMax, nZs)

# Halo Model
dictnumber = 21
hod_name = "unWISE"+unWISEcol

hcos = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', mdef='vir', concmode='BHATTACHARYA', unwise_color=unWISEcol, choose_dict=dictnumber)
hcos.add_hod(name=hod_name)
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

dvols  = get_volume_conv(chis, Hz)
PzkLin = hcos._get_matter_power(zs, ks, nonlinear=False)
Pzell  = get_fourier_to_multipole_Pkz(zs, ks, chis, ellMax0, PzkLin)
Pzell0 = Pzell.transpose(1,0)
print('Done turning into multipoles.')

path_params0 = np.asarray([MA, nZs, zMin, zMax, ellMax0, rscale])

dothis = False
if dothis:
    rcross = get_rcross_per_halo(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, pick_prof, name=name)
    ucosth, angs = get_halo_skyprofile(zs, chis, rcross)

    prob = conv_prob(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=rscale, name=name)
    #prob = conv_prob_flat(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=rscale, name=name)

    utheta = prob_theta(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=rscale, name=name)

    print('Computing multipole expansion of angular probability u.')
    partial_u = functools.partial(get_uell0, angs, ucosth)
    with ProcessPoolExecutor(num_workers) as executor:
        uell0 = list(executor.map(partial_u, ells, chunksize=chunksize))

    prob00 = prob * utheta * uell0[0]
    avtau, dtaudz = get_avtau(zs, ms, nzm, dvols, prob00)

    np.save(data_path(*path_params0), [rcross, prob, utheta, avtau, dtaudz, uell0])

    # For polarization, the number of resonances per halo is two, as L_domains < L_halo and in+out events are uncorrelated
    # so we need utheta**2.
    rcross, prob, utheta, avtau, dtaudz, uell0 = np.load(data_path(*path_params0))
    probell_temp = (prob * utheta)[None,...] * uell0

    # Z_ell has dimensions Nz, Nm, Nell
    ct = np.sqrt((4.*np.pi)/(2*ells+1))
    zell_tau = (probell_temp[:ellMax0]) * ct[:, None, None]

    # Assemble power spectra
    int_uell_1h = np.trapz(nzm[None,...] * zell_tau**2.               , ms, axis=-1)
    int_uell_2h = np.trapz(nzm[None,...] * biases[None,...] * zell_tau, ms, axis=-1)

    Cl1h  = np.trapz(dvols[None,:] * int_uell_1h                     , zs, axis=1)
    Cl2h  = np.trapz(dvols[None,:] * np.abs(int_uell_2h)**2. * Pzell0, zs, axis=1)
    np.save(cl_data_tautau_path(*path_params0), [Cl1h, Cl2h])

    Cl1h, Cl2h = np.load(cl_data_tautau_path(*path_params0))
    np.save(ClTT0_path(*path_params0), (Cl1h + Cl2h) * TCMB**2.)

dothis=True
if dothis:
    print('Importing dark screening in temperature data.')
    rcross, prob, utheta, avtau, dtaudz, uell0 = np.load(data_path(*path_params0))

    for ellMax in [6000]:
        dopowc = True
        dobisp = False
        dobispconstr = False

        l0Max, l1Max, l2Max = ellMax, ellMax, ellMax
        ells0 = np.arange(ellMax0)
        ells  = np.arange(ellMax)
        print('Compute screening of polarization from axion. ellMAX = ', ellMax)

        # To compute the polarization screening, I am using the temperature dark screening integrand per halo
        # so I need to multiply by 3 in each halo
        # For polarization, the number of resonances per halo is one, as L_domains < L_halo and in+out events are uncorrelated
        probell_pol = 3. * (prob[None,...] * uell0)[:ellMax0,...] * np.sqrt((4.*np.pi)/(2*ells0+1))[:, None, None]
        zell_pol    = np.abs( utheta[None,...] * np.abs(probell_pol)**2. )

        for Npc in [5., 1., 10.]:
            print('Doing', Npc, 'kiloparsecs per magnetic domain with ellMax=', ellMax)
            ellshort = np.array([0.] + np.geomspace(2, ellMax-1, 51).tolist())
            ellshort = np.array([int(ii) for ii in ellshort])
            ellshort = np.array(list(dict.fromkeys(ellshort)))

            # polarization window function; function of l1, z
            Clpol = get_gaussian_pol_1kp(zs, chis, ellMax0, Npc)

            if dopowc:
                # one-halo axion screening; function of l2, z
                dCl1hdz = dvols[None,:] * np.trapz(nzm[None,...] * zell_pol, ms, axis=-1)

                Clmix = np.trapz(dCl1hdz[:ellMax,None,:] * Clpol[None,:ellMax,:], zs, axis=-1)
                scrEE = get_scrCLs_pol_axion(l0Max, l1Max, l2Max, ellshort, Clmix, TCMB)

                path_params = np.asarray([MA, nZs, zMin, zMax, ellMax, rscale])
                np.save(fullscr_polaxion_tautau_path(*path_params, Npc), scrEE)
                print('Saved polarization auto.')

            if dobisp:
                # This is the galaxy template defined by whether there is a central galaxy at a point or not; function of z2 and m2
                Ncs  = hcos.hods[hod_name]['Nc']
                dndz = get_dndzHOD(zs, pathdndz, dvols)
                N_gtot = np.trapz(dndz, zs, axis=0)
                W_g = dndz/N_gtot/dvols
                ngalcentrals = np.trapz(nzm * Ncs[None,:], ms, axis=-1)
                uellgcen = W_g[:,None] * Ncs[None,:] / ngalcentrals[:,None]

                # The dark screening integrand; function of L and z
                integrBB = np.trapz(nzm[None,...] * biases[None,...] * zell_pol, ms, axis=-1)

                # The central gal integrand times the linear matter power; function of ell'' and z
                integrCen = np.trapz(nzm * biases * uellgcen, ms, axis=-1)
                dCellhh   = dvols[None,:] * integrCen[None,:] * Pzell0

                # multiplying by magnetic field domain shape; integrate z away; shape ell'', l_1, l_2
                Blitgr = np.trapz(dCellhh[ellshort,None,None,:] * integrBB[None, :ellMax, None, :] * Clpol[None,None,ellshort,:], zs, axis=-1)

                # Keeping ell'' fixed, sum over l1, l2 weighting by appropriate 3j symbols
                # It returns the linear matter PS ell on position 0 and the other ell on position 1
                Bispec = get_scrBLs_polaxion_gal(l0Max, l1Max, l2Max, ellshort, Blitgr, TCMB)

                path_params = np.asarray([MA, nZs, zMin, zMax, ellMax, rscale])
                np.save(fullscr_reducedbisp_tautau_path(*path_params, Npc) + '_unWISE.npy', Bispec)
                print('Saved red bisp MA, npc', MA, Npc)

            if dobispconstr:
                for eid, (expname, experiment) in enumerate(zip(['Planck', 'CMBS4'], [Planck, CMBS4])):
                    if eid!=0: continue

                    baseline = ghztoev(145)
                    units  = xov(baseline) * baseline
                    fsky = 0.4

                    if expname == 'Planck':
                        mm, mmm = 4, 3000
                    elif expname == 'CMBS4':
                        mm, mmm = 4, 6000

                    Bispec      = np.load(fullscr_reducedbisp_tautau_path(MA, nZs, zMin, zreio, ellMax, rscale, Npc)+'_unWISE.npy')
                    leftover    = np.load(ILCnoisePS_path(MA, nZs, zMin, zreio, ellMax0, expname))
                    Cell_galgal = np.load(cl_data_galgal_path(nZs, zMin, zreio, ellMax0, name=name, galcol=unWISEcol, dictn=dictnumber) + 'centrals_only_unWISE.npy')

                    xg, yg = np.meshgrid(ells, ells, indexing='ij', sparse=True)
                    f = RegularGridInterpolator((ellshort, ells), Bispec)
                    Bispec = f((xg, yg)) * units**2.

                    constr = get_bispectrum_constraint(fsky, mm, mmm, ellMax, leftover[2], Cell_galgal, Bispec)
                    np.save(bispectrum_constraint(MA, nZs, zMin, zreio, ellMax, Npc, expname, unwi=True), constr)
                    print('Saved', expname, Npc, MA, constr)

print('Done maind, MA', maind, MA)
