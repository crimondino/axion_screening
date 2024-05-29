import hmvec as hm
import numpy as np
import scipy as scp
from scipy.special import eval_legendre, legendre, spherical_jn
import itertools
import wigner
from sympy.physics.wigner import wigner_3j
import time
from scipy import interpolate
from itertools import cycle
from math import atan2,degrees,lgamma 
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp2d,interp1d
import scipy.interpolate as si

from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator
import random
import seaborn as sns

from params import *
#from plotting import *

############### Compute Polarization Screening ###########################

def get_dndz(path, dvols, zs):
    dndz_data= np.transpose(np.loadtxt(path, dtype=float))
    zsHOD    = dndz_data[0,:]
    dndz     = np.interp(zs, dndz_data[0,:], dndz_data[1,:])
    N_gtot   = np.trapz(dndz, zs, axis=0)
    W_g      = dndz/N_gtot/dvols
    return dndz, zs, N_gtot, W_g, dndz_data[0,:], dndz_data[1,:]

def get_gaussian_pol_1kp(zs, chis, ellMax, Npc=1.):
    rchis = chis*aa(zs)[None,:]

    # kiloparsecs; Npc = number of kpc
    lendom = 1e-3 * Npc

    sigma   = lendom / rchis
    sigmasq = sigma**2. / (8. * np.log(2.))

    ells = np.arange(ellMax)[:,None]

    expo = - ells * (ells + 1.) * sigmasq / 2.

    axEEell = 1./15. * 2.*np.pi*sigmasq * np.exp(expo)
    axBBell = 1./15. * 2.*np.pi*sigmasq * np.exp(expo)
    return sigma, axEEell, axBBell

def get_scrCLs_pol_axion(l0Max, l1Max, l2Max, PPCl, TCMB):
    l0List   = np.arange(   l0Max)
    ell1_CMB = np.arange(2, l1Max)
    ell2_scr = np.arange(2, l2Max)

    every_pair = np.asarray(list(itertools.product(ell1_CMB, ell2_scr)))
    allcomb = len(every_pair)
    nums = np.array(np.linspace(0, allcomb, 10), dtype=int).tolist()

    scrEE, scrBB, wig220list = np.zeros((3, l0Max))
    for ind, (l1,l2) in enumerate(every_pair):
        if ind in nums: print(ind, 'out of', allcomb, nums.index(ind)+1)

        wig220list[:] = 0.

        norm = (2.*l1+1.)*(2.*l2+1.)/(4.*np.pi)

        w220   = wigner.wigner_3jj(l1, l2, 2, 0)
        cm, dm = max(2, int(w220[0])), min(int(w220[1]), l0Max-1)
        l220   = np.arange(cm, dm+1)
        cw, dw = int(cm - w220[0]), int(dm - w220[0])

        wig220list[l220] = np.abs(w220[2][cw:dw+1])**2.

        # index l2 is dark screening (spin 0), index l1 is the polarization (spin 2)
        mix    = norm * PPCl[l2, l1] * wig220list
        Jell   = l0List+l1+l2
        delte  = 0.5*(1. + (-1.)**Jell)
        delto  = 0.5*(1. - (-1.)**Jell)

        scrEE += mix * delte
        scrBB += mix * delto

    Tfact = TCMB**2./(4.*np.pi)
    return l0List, scrEE*Tfact, scrBB*Tfact


def get_scrBLs_polaxion_gal(l0Max, l1Max, l2Max, Blmixx):
    l0List   = np.arange(   l0Max)
    ell1_CMB = np.arange(2, l1Max)
    ell2_scr = np.arange(2, l2Max)

    every_pair = np.asarray(list(itertools.product(ell1_CMB, ell2_scr)))
    allcomb = len(every_pair)
    nums    = np.array(np.linspace(0, allcomb, 10), dtype=int).tolist()

    lshort = np.shape(Blmixx)[0]
    scrbispEE, scrbispBB = np.zeros((2, lshort, l0Max))
    wig220  = np.zeros(l0Max)
    for ind, (l1,l2) in enumerate(every_pair):

        if ind in nums: print(ind, 'out of', allcomb, nums.index(ind)+1)

        norm = (2.*l1+1.)*(2.*l2+1.)/(4.*np.pi)

        w220   = wigner.wigner_3jj(l1, l2, 2, 0)
        cm, dm = max(2, int(w220[0])), min(int(w220[1]), l0Max-1)
        l220   = np.arange(cm, dm+1)
        cw, dw = int(cm - w220[0]), int(dm - w220[0])

        wig220[:] = 0.
        wig220[l220] = np.abs(w220[2][cw:dw+1])**2.

        # index l2 is dark screening (spin 0), index l1 is the polarization (spin 2)
        mix    = norm * Blmixx[:, l2, l1][:,None] * wig220[None, :]
        Jell   = l0List+l1+l2
        delte  = 0.5*(1. + (-1.)**Jell)
        delto  = 0.5*(1. - (-1.)**Jell)

        scrbispEE += mix * delte[None,:]
        scrbispBB += mix * delto[None,:]
    return scrbispEE, scrbispBB


def get_scrBLs_approximation_polaxion_gal(l1Max, Blmixx):
    ELLlist= np.arange(2, l1Max)

    lshort = np.shape(Blmixx)[0]
    scrbispEE, scrbispBB = np.zeros((2, lshort))

    for ELL in ELLlist:
        ELL = int(ELL)
        for l2 in [ELL-3, ELL-1, ELL+1, ELL+3]:
            if l2 >= l1Max: continue

            a = wigner_3j(3, ELL, l2, -2, 2, 0)
            symb = (2.*ELL+1.)*(2.*l2+1.)/(4.*np.pi) * np.abs(float(a))**2.

            scrbispEE += symb * Blmixx[:, l2, ELL]

        for l2 in [ELL-2, ELL, ELL+2]:
            if l2 >= l1Max: continue

            a = wigner_3j(3, ELL, l2, -2, 2, 0)
            symb = (2.*ELL+1.)*(2.*l2+1.)/(4.*np.pi) * np.abs(float(a))**2.

            scrbispBB += symb * Blmixx[:, l2, ELL]
    return scrbispEE, scrbispBB





def get_wigner220(l0Max, l1Max, l2Max):
    l0List   = np.arange(   l0Max)
    ell1_CMB = np.arange(2, l1Max)
    ell2_scr = np.arange(2, l2Max)

    every_pair = np.asarray(list(itertools.product(ell1_CMB, ell2_scr)))
    allcomb = len(every_pair)
    nums    = np.array(np.linspace(0, allcomb, 10), dtype=int).tolist()

    wig220  = np.zeros(l0Max)
#    save_wig220_EE = np.zeros((l0Max, l1Max, l2Max))
#    save_wig220_BB = np.zeros((l0Max, l1Max, l2Max))
    save_wig220_tot = np.zeros((l0Max, l1Max, l2Max))
    for ind, (l1,l2) in enumerate(every_pair):

        if ind in nums: print(ind, 'out of', allcomb, nums.index(ind)+1)

        norm = (2.*l1+1.)*(2.*l2+1.)/(4.*np.pi)

        w220   = wigner.wigner_3jj(l1, l2, 2, 0)
        cm, dm = max(2, int(w220[0])), min(int(w220[1]), l0Max-1)
        l220   = np.arange(cm, dm+1)
        cw, dw = int(cm - w220[0]), int(dm - w220[0])

        wig220[:] = 0.
        wig220[l220] = w220[2][cw:dw+1]

        mix    = norm * wig220**2.
#        Jell   = l0List+l1+l2
#        delte  = 0.5*(1. + (-1.)**Jell)
#        delto  = 0.5*(1. - (-1.)**Jell)

#        save_wig220_EE[:,l1,l2] = mix * delte
#        save_wig220_BB[:,l1,l2] = mix * delto
        save_wig220_tot[:,l1,l2]= mix

    return save_wig220_tot#, save_wig220_EE, save_wig220_BB


############### COMPUTE ANGULAR POWER SPECTRA ###########################

def get_rcross_per_halo(zs, ms, rs, rvir, rhocritz, deltav, cs, MA, pick_prof, name='battagliaAGN'):
    if pick_prof:
        return get_rcross_per_halo_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, MA, name)
    else:
        print('Error: NFW profile not implemented')
        return None

def get_rcross_per_halo_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, MA, name='battagliaAGN'):
    """ Compute crossing radius of each halo
    i.e. radius where plasma mass^2 = dark photon mass^2
    Find the index of the radius array where plasmon mass^2 = dark photon mass^2 """

    m200critz, r200critz = get_200critz(zs, ms, cs, rhocritz, deltav)

    rcross_res = np.zeros((len(zs), len(ms)))
    for i_z, z in enumerate(zs):
        for i_m, m in enumerate(ms):
            func = lambda x: np.abs(get_gas_profile(x, z, m200critz[i_z, i_m], r200critz[i_z, i_m], rhocritz[i_z], name=name) * conv/MA**2. - 1.)
            rcross_res[i_z, i_m] = fsolve(func, x0=rs[0])
    return rcross_res


def conv_prob(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=False, name='battagliaAGN'):
    if pick_prof:
        return conv_prob_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, logB_interp_list, mass_bins, rad_bins, name)
    else:
        print('Error: NFW profile not implemented')
        return None

def conv_prob_flat(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=False, name='battagliaAGN'):
    if pick_prof:
        return conv_prob_flat_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, logB_interp_list, mass_bins, rad_bins, name)
    else:
        print('Error: NFW profile not implemented')
        return None


def conv_prob_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, name='battagliaAGN'):
    """Conversion probability including the B field profile"""
    m200, r200 = get_200critz(zs, ms, cs, rhocritz, deltav)
    drhodr = get_deriv_gas_profile(rcross, zs[:,None], m200, r200, rhocritz[:,None], name=name)
    dmdr   = np.abs(conv*drhodr)
    omgz   = (1.+zs[:,None])# * omega0 but we want to keep frequency dependence separate
    uang   = 2.*np.heaviside(rvir - rcross, 0.5)

    bprof = get_B_rcross(zs, ms, m200, r200, rhocritz, rcross, logB_interp_list, mass_bins, rad_bins)
    units = np.pi/3. * gauss2evsq(1.)**2. * mpcEVinv
    return units * uang * bprof**2. * omgz / dmdr

def conv_prob_flat_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, name='battagliaAGN'):
    """Conversion probability including the B field profile"""
    m200, r200 = get_200critz(zs, ms, cs, rhocritz, deltav)
    drhodr = get_deriv_gas_profile(rcross, zs[:,None], m200, r200, rhocritz[:,None], name=name)
    dmdr   = np.abs(conv*drhodr)
    omgz   = (1.+zs[:,None])# * omega0 but we want to keep frequency dependence separate
    uang   = 2.*np.heaviside(rvir - rcross, 0.5)

    bprof = get_B_rcross_flat(zs, ms, m200, r200, rhocritz, rcross, logB_interp_list, mass_bins, rad_bins)
    units = np.pi/3. * gauss2evsq(1.)**2. * mpcEVinv
    return units * uang * bprof**2. * omgz / dmdr


def get_B_rcross(zs, ms, m200c, r200c, rhocritz, rcross, logB_interp_list, mass_bins, rad_bins):
    rcross_ratio = rcross/r200c
    Brcross = np.zeros((len(zs), len(ms)))

    ms_ind = np.digitize(m200c[0, :], mass_bins)
    ms_ind[ms_ind == len(logB_interp_list)] = len(logB_interp_list)-1.

    for i_m in range(len(ms)):
        for i_z, z_val in enumerate(zs):        
            if rcross_ratio[i_z, i_m] < rad_bins[0]:
                Brcross[i_z, i_m] = ( get_pth_profile(rcross_ratio[i_z, i_m]*r200c[i_z, i_m], z_val, m200c[i_z, i_m], r200c[i_z, i_m], rhocritz[i_z]) /
                                      get_pth_profile(rad_bins[0]*r200c[i_z, i_m], z_val, m200c[i_z, i_m], r200c[i_z, i_m], rhocritz[i_z]) ) * 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rad_bins[0]] )   
            else:
                Brcross[i_z, i_m] = 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rcross_ratio[i_z, i_m] ] )     
    return Brcross

def get_B_rcross_flat(zs, ms, m200c, r200c, rhocritz, rcross, logB_interp_list, mass_bins, rad_bins):
    rcross_ratio = rcross/r200c
    Brcross = np.zeros((len(zs), len(ms)))

    ms_ind = np.digitize(m200c[0, :], mass_bins)
    ms_ind[ms_ind == len(logB_interp_list)] = len(logB_interp_list)-1

    for i_m in range(len(ms)):
        for i_z, z_val in enumerate(zs):        
            if rcross_ratio[i_z, i_m] < rad_bins[0]:
                Brcross[i_z, i_m] = 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rad_bins[0]] )   
            else:
                Brcross[i_z, i_m] = 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rcross_ratio[i_z, i_m] ] )     
    return Brcross


def get_volume_conv(chis, Hz):
    # Volume of redshift bin divided by Hubble volume
    # Chain rule factor when converting from integral over chi to integral over z
    return chis**2. / Hz

def get_200critz(zs, ms, cs, rhocritz, deltav):
    delta_rhos1 = deltav*rhocritz
    delta_rhos2 = 200.*rhocritz
    m200critz = hm.mdelta_from_mdelta(ms, cs, delta_rhos1, delta_rhos2)
    r200critz = hm.R_from_M(m200critz, rhocritz[:,None], delta=200.)
    return m200critz, r200critz

def get_halo_skyprofile(zs, chis, rcross):
    # get bounds of each regime within halo
    rchis = chis*aa(zs)
    fract = (rcross/rchis[:,None])[None,...]

    listincr = 1. - np.geomspace(1e-3, 1., 41)
    listincr = np.asarray([1.] + listincr.tolist())[::-1]
    angs = listincr[:,None,None] * fract

    ucosth = (1.-(angs/fract)**2.)**(-0.5)
    ucosth[angs == fract] = 0.
    return ucosth, angs


def get_u00(zs, chis, rcross):
    # this gives the analytical result for the monopole
    rchis = chis*aa(zs)
    fract = (rcross/rchis[:,None])
    return fract**2./2.

def get_uell0(angs, ucosth, ell):
    # this returns the analytical approximation for low ell
    # or numerical result for higher multipoles
 
    uL0 = np.zeros(angs[0].shape)

    approx = ell < 0.1/angs[-1, :, :] # indices for which we can use the approximation

    uL0[approx] = 2.*np.pi * (angs[-1, :, :][approx])**2.

    # angular function u(theta) is projected into multipoles
    cos_angs = np.cos(angs[:, ~approx])
    Pell     = eval_legendre(ell, cos_angs)
    integr   = Pell * np.sin(angs[:, ~approx]) * ucosth[:, ~approx]
    uL0[~approx] = 2.*np.pi * np.trapz(integr, angs[:,~approx], axis=0)

    if ell%100==0: print(ell)
    return uL0 * ((4.*np.pi) / (2.*ell+1.))**(-0.5)



def get_gas_profile(rs, zs, m200, r200, rhocritz, name='battagliaAGN'):
    if name!='battagliaAGN':
        print('This gas profile is not implemented. Using battagliaAGN instead.')
    rho0, alpha, beta, gamma, xc = battagliaAGN(m200, zs)

    rho = rhocritz * rho0
    x = rs/r200/xc
    expo = -(beta+gamma)/alpha # gamma sign must be opposite from Battaglia/ACT paper; typo
    return rho * (x**gamma) * ((1.+x**alpha)**expo)

def get_deriv_gas_profile(rs, zs, m200, r200, rhocritz, name='battagliaAGN'):
    if name!='battagliaAGN':
        print('This gas profile is not implemented. Using battagliaAGN instead.')
    rho0, alpha, beta, gamma, xc = battagliaAGN(m200, zs)

    rho = rhocritz * rho0
    x = rs/r200/xc
    expo = -(alpha+beta+gamma)/alpha
    
    drhodr = rho * (x**gamma) * (1. + x**alpha)**expo * (gamma - x**alpha * beta) / rs

    if hasattr(rs, "__len__"): drhodr[rs==0.] = 0.
    elif rs==0: drhodr = 0.
    return drhodr

def get_avtau(zs, ms, nzm, dvol, prob00):
    # Average optical depth per redshift bin
    dtaudz = np.trapz(nzm * prob00, ms, axis=-1) * dvol * 4*np.pi
    avtau  = np.trapz(dtaudz, zs, axis=0)
    return avtau, dtaudz

def get_dtauell(ms, nzm, dvol, biases, probell):
    # integrand in 2-halo numerator
    return np.trapz(biases[None,...]*nzm[None,...]*probell, ms, axis=2) * dvol[None,:]

def get_Celldtaudtau_1h(zs, ms, ks, nzm, dvol, probell, ellMax):
    # The 1-halo term
    ells = np.arange(ellMax)
    Cl1hdz = np.trapz(nzm[None,...] * np.abs(probell)**2., ms, axis=2)
    Cl1h = np.trapz(Cl1hdz * dvol[None,:], zs, axis=1)
    return Cl1h * (4.*np.pi) / (2.*ells+1.)

def get_dCelldtaudtaudz_1h(zs, ms, ks, nzm, dvol, probell, ellMax):
    # The 1-halo term but nothing is integrated
    ells = np.arange(ellMax)[:,None]
    fact = (4.*np.pi) / (2.*ells+1.)
    Cl1hdz = np.trapz(nzm[None,...] * np.abs(probell)**2., ms, axis=2) * dvol[None,:]
    return Cl1hdz * fact

def get_dCelldtaudtaudzdm_1h(zs, ms, ks, nzm, dvol, probell, ellMax):
    # The 1-halo term but nothing is integrated
    ells = np.arange(ellMax)[:,None,None]
    fact = (4.*np.pi) / (2.*ells+1.)
    # there's already a factor of vol z2 in dCellhhdzdm
    Cl1hdzdm = nzm[None,...] * np.abs(probell)**2.
    return Cl1hdzdm * fact

def get_Celldtaudtau_2h(zs, ms, ks, chis, PzkLinz1z2, dtauell, ell):
    jn     = spherical_jn(ell, chis[:,None]*ks[None,:])
    jnsq   = jn[:,None,:]*jn[None,...]
    kssq   = ks**2.

    dtausq = dtauell[ell,:,None]*dtauell[ell,None,:]
    Cl2hdz = 2./np.pi*dtausq * np.trapz(kssq[None,None,:] * PzkLinz1z2 * jnsq, ks, axis=-1)
    Cl2h   = np.trapz(np.trapz(Cl2hdz, zs[None,:], axis=-1), zs, axis=-1)

    if ell%100==0: print(ell)
    return Cl2h * (4.*np.pi) / (2.*ell+1.)

def get_dCelllin(zs, ms, ks, chis, nzm, dvols, biases, Nczm, PzkLinz1z2, ell):
    jn   = spherical_jn(ell, chis[:,None]*ks[None,:])
    integ= (ks**2.)[None,None,:] * PzkLinz1z2 * jn[:,None,:]*jn[None,...]

    Clinear = np.trapz(integ, ks, axis=-1)

    multi   = biases * nzm * dvols[:,None]
    multisq = multi[:,:,None,None] * multi[None,None,:,:] * Nczm[None,None,:,:]

    dCellhhdzdm = np.trapz(np.trapz(Clinear[:,None,:,None] * multisq, ms, axis=-1), zs, axis=-1)
 
    # first bit comes from Plin, second factor from integrating out the Ylm(0,0)
    return dCellhhdzdm * 2./np.pi * np.sqrt((2. * ell + 1.) / (4. * np.pi))





def w000(Ell, ell0, ell1, ell2):
    # fast wigner 3j with m1 = m2 = m3 = 0
    g = Ell/2.
    w = np.exp(0.5*(lgamma(2.*g-2.*ell0+1.)+lgamma(2.*g-2.*ell1+1.)+lgamma(2.*g-2.*ell2+1.)-lgamma(2.*g+2.))+lgamma(g+1.)-lgamma(g-ell0+1.)-lgamma(g-ell1+1.)-lgamma(g-ell2+1.))
    return w * (-1.)**g


def get_scrCLs(l0Max, l1Max, l2Max, CMB, DPCl):
    l0List   = np.arange(   l0Max)
    ell1_CMB = np.arange(2, l1Max)
    ell2_scr = np.arange(2, l2Max)

    TTCl, EECl, BBCl, TECl = CMB[:,0], CMB[:,1], CMB[:,2], CMB[:,3]

    every_pair = np.asarray(list(itertools.product(ell1_CMB, ell2_scr)))
    allcomb = len(every_pair)
    nums    = np.array(np.linspace(0, allcomb, 10), dtype=int).tolist()

    scrTT, scrEE, scrBB, scrTE = np.zeros(l0Max), np.zeros(l0Max), np.zeros(l0Max), np.zeros(l0Max)
    wig000  = np.zeros(l0Max)
    wig220  = np.zeros(l0Max)
    for ind, (l1,l2) in enumerate(every_pair):

        if ind in nums: print(ind, 'out of', allcomb, nums.index(ind)+1)

        norm = (2.*l1+1.)*(2.*l2+1.)/(4.*np.pi)

        w000   = wigner.wigner_3jj(l1, l2, 0, 0)
        am, bm = max(2, int(w000[0])), min(int(w000[1]), l0Max-1)
        l000   = np.arange(am, bm+1)
        aw, bw = int(am - w000[0]), int(bm - w000[0])

        wig000[:] = 0.
        wig000[l000] = w000[2][aw:bw+1]

        scrTT += norm * DPCl[l2] * TTCl[l1] * wig000**2.

        w220   = wigner.wigner_3jj(l1, l2, 2, 0)
        cm, dm = max(2, int(w220[0])), min(int(w220[1]), l0Max-1)
        l220   = np.arange(cm, dm+1)
        cw, dw = int(cm - w220[0]), int(dm - w220[0])

        wig220[:] = 0.
        wig220[l220] = w220[2][cw:dw+1]

        scrTE += norm * DPCl[l2] * TECl[l1] * wig000 * wig220

        mix    = norm * DPCl[l2] * EECl[l1] * wig220**2.
        Jell   = l0List+l1+l2
        delte  = 0.5*(1. + (-1.)**Jell)
        delto  = 0.5*(1. - (-1.)**Jell)

        scrEE += mix * delte
        scrBB += mix * delto

    return l0List, scrTT, scrEE, scrBB, scrTE


def get_Tmonopole_screeningPS(l0Max, l2Max, CMB, DPCl):
    l0List   = np.arange(   l0Max)
    ell2_scr = np.arange(2, l2Max)
    TTC0 = CMB[0,0]

    scrTT = np.zeros(l0Max)
    wig000 = np.zeros(l0Max)
    for l2 in ell2_scr:
        norm = (2.*l2+1.)*(4.*np.pi)**-1.
        w000 = wigner.wigner_3jj(0, l2, 0,  0)
        am, bm  = max(2, int(w000[0])), min(int(w000[1]), l0Max-1)
        l0list  = np.arange(am, bm+1)
        aw, bw  = int(am - w000[0]), int(bm - w000[0])
        wig000[:]   = 0.
        wig000[l0list] = w000[2][aw:bw+1]
        scrTT += norm * DPCl[l2] * TTC0 * wig000**2.
    return l0List, scrTT


def bispectrum_Tdsc_Tsc_Tsc(l0Max, l1Max, l2Max, T0, ClTT, Cltautaurei, NICLdscdsc, NICLscsc):
    l1List = np.arange(2, l1Max)
    l2List = np.arange(2, l2Max)

    all_possible_pairs = np.asarray(list(itertools.product(l1List, l2List)))
    sumlist = np.zeros(len(all_possible_pairs))

    for ind, (l1,l2) in enumerate(all_possible_pairs):
        wig3j = wigner.wigner_3jj(l2, l1, 0,  0)

        am, bm  = max(2, int(wig3j[0])), min(int(wig3j[1]), l0Max-1)
        l0list  = np.arange(am, bm+1)
        aw, bw  = int(am - wig3j[0]), int(bm - wig3j[0])
        wig000  = wig3j[2][aw:bw+1]

        norm   = (wig000 * T0)**2. * (2.*l0list+1.)
        numer  = norm * ((ClTT[l0list] + ClTT[l1]) * Cltautaurei[l2])**2.
        denom  = NICLdscdsc[l2] * NICLscsc[l0list] * NICLscsc[l1]

        sumlist[ind] = np.sum(numer / denom)
    return np.sum(sumlist)

def bispectrum_Tdsc_Esc_Bsc(l0Max, l1Max, l2Max, T0, ClEE, Cltautaurei, NICLdscdsc, NICLEEscsc, NICLBBscsc):
    l1List = np.arange(2, l1Max)
    l2List = np.arange(2, l2Max)

    all_possible_pairs = np.asarray(list(itertools.product(l1List, l2List)))
    sumlist = np.zeros(len(all_possible_pairs))

    for ind, (l1,l2) in enumerate(all_possible_pairs):
        wig3j = wigner.wigner_3jj(l2, l1, 0,  2)

        am, bm  = max(2, int(wig3j[0])), min(int(wig3j[1]), l0Max-1)
        l0list  = np.arange(am, bm+1)
        aw, bw  = int(am - wig3j[0]), int(bm - wig3j[0])
        wig220  = wig3j[2][aw:bw+1]

        Jell   = l0list+l1+l2
        delto  = 0.5*(1. - (-1.)**Jell)

        norm   = delto * (wig220 * T0)**2. * (2.*l0list+1.)
        numer  = norm * ((ClEE[l0list] + ClEE[l1]) * Cltautaurei[l2])**2.
        denom  = NICLdscdsc[l2] * NICLBBscsc[l0list] * NICLEEscsc[l1]

        sumlist[ind] = np.sum(numer / denom)
    return np.sum(sumlist)


########## Noises ###########



########## Covariance Matrices + Fisher forecasting ###########

def sigma_screening(epsilon4, ellmin, ellmax, screening, leftover):
    #print('Full ', np.shape(leftover), np.shape(screening))
    ClTTNl = epsilon4 * screening[:, 0] + leftover[:, 0]
    ClEENl = epsilon4 * screening[:, 1] + leftover[:, 1]
    ClBBNl = epsilon4 * screening[:, 2] + leftover[:, 2]

    dClTTde4 = screening[:, 0]
    dClEEde4 = screening[:, 1]
    dClBBde4 = screening[:, 2]

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCov    = np.diag([ClTTNl[el], ClEENl[el], ClBBNl[el]])
        CCovInv = np.linalg.inv(CCov)
        dCovde4 = np.diag([dClTTde4[el], dClEEde4[el], dClBBde4[el]])
        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde4@CCovInv@dCovde4)
    return 0.21 * (np.sum(TrF)**(-0.25))**0.5

def sigma_screening_TTonly(epsilon4, ellmin, ellmax, screening, leftover):
    #print('TT only', np.shape(leftover), np.shape(screening))
    ClTTNl   = epsilon4 * screening[:ellmax, 0] + leftover[:ellmax, 0]
    dClTTde4 = screening[:ellmax, 0]

    ells = np.arange(ellmax)
    TrF  = 0.5*(2.*ells+1.)*(dClTTde4/ClTTNl)**2.
    return 0.21 * (np.sum(TrF[ellmin:ellmax])**(-0.25))**0.5

def sigma_screening_BBonly(epsilon4, ellmin, ellmax, screening, leftover):
    #print('BB only', np.shape(leftover), np.shape(screening))
    ClBBNl   = epsilon4 * screening[:ellmax, 2] + leftover[:ellmax, 2]
    dClBBde4 = screening[:ellmax, 2]

    ells = np.arange(ellmax)
    TrF  = 0.5*(2.*ells+1.)*(dClBBde4/ClBBNl)**2.
    return 0.21 * (np.sum(TrF[ellmin:ellmax])**(-0.25))**0.5

def sigma_screeningVtemplate(TCMB, ep2, ellmin, ellmax, cltauscreening, leftover, templateclgalgal):
    ClTTNl      = ep2**2. * cltauscreening[:, 0] + leftover[:, 0]
    Clττ        = templateclgalgal
    ClTτscr     = ep2 * templateclgalgal * TCMB/np.sqrt(4.*np.pi)
    dClTTde2    = 2.*ep2 * cltauscreening[:, 0]
    dClTτscrde2 = templateclgalgal * TCMB/np.sqrt(4.*np.pi)

    TrF = np.empty(ellmax)
    for el in range(ellmin, ellmax):
        CCov = np.asarray([[Clττ[el]   , ClTτscr[el]],\
                           [ClTτscr[el], ClTTNl[el] ]])
        CCovInv = np.linalg.inv(CCov)
        dCovde2 = np.asarray([[0.             , dClTτscrde2[el]],\
                              [dClTτscrde2[el], dClTTde2[el]   ]])

        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde2@CCovInv@dCovde2)
    return 0.35 * np.sum(TrF[ellmin:ellmax])**(-0.25)

def sigma_screeningVunWISE(TCMB, ep2, ellmin, ellmax, cltauscreening, leftover, clgaltau, clgalgal):
    ClTTNl      = ep2**2. * cltauscreening[:, 0] + leftover[:, 0]
    Clττ        = clgalgal
    ClTτscr     = ep2 * clgaltau
    dClTTde2    = 2.*ep2 * cltauscreening[:, 0]
    dClTτscrde2 = clgaltau

    TrF = np.empty(ellmax)
    for el in range(ellmin, ellmax):
        CCov = np.asarray([[Clττ[el]   , ClTτscr[el]],\
                           [ClTτscr[el], ClTTNl[el] ]])
        CCovInv = np.linalg.inv(CCov)
        dCovde2 = np.asarray([[0.             , dClTτscrde2[el]],\
                              [dClTτscrde2[el], dClTTde2[el]   ]])

        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde2@CCovInv@dCovde2)
    return 0.35 * np.sum(TrF[ellmin:ellmax])**(-0.25)

def sigma_screeningVtemplate(TCMB, ep2, ellmin, ellmax, cltauscreening, leftover, templateclgalgal):
    ClTTNl      = ep2**2. * cltauscreening[:, 0] + leftover[:, 0]
    ClEENl      = ep2**2. * cltauscreening[:, 1] + leftover[:, 1]
    ClBBNl      = ep2**2. * cltauscreening[:, 2] + leftover[:, 2]

    Clττ        = templateclgalgal
    ClTτscr     = ep2 * templateclgalgal * TCMB/np.sqrt(4.*np.pi)

    dClTTde2    = 2.*ep2 * cltauscreening[:, 0]
    dClEEde2    = 2.*ep2 * cltauscreening[:, 1]
    dClBBde2    = 2.*ep2 * cltauscreening[:, 2]

    dClTτscrde2 = templateclgalgal * TCMB/np.sqrt(4.*np.pi)
    dClEτscrde2 = templateclgalgal * TCMB/np.sqrt(4.*np.pi)
    dClBτscrde2 = templateclgalgal * TCMB/np.sqrt(4.*np.pi)

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCov = np.asarray([[Clττ[el]   , ClTτscr[el], 0.        , 0.        ],\
                           [ClTτscr[el], ClTTNl[el] , 0.        , 0.        ],\
                           [0.         , 0.         , ClEENl[el], 0.        ],\
                           [0.         , 0.         , 0.        , ClBBNl[el]]])
        CCovInv = np.linalg.inv(CCov)
        dCovde2 = np.asarray([[0.             , dClTτscrde2[el], dClEτscrde2[el], dClBτscrde2[el]],\
                              [dClTτscrde2[el], dClTTde2[el]   , 0.             , 0.             ],\
                              [dClEτscrde2[el], 0.             , dClEEde2[el]   , 0.             ],\
                              [dClBτscrde2[el], 0.             , 0.             , dClBBde2[el]   ]])

        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde2@CCovInv@dCovde2)
    return 0.35 * np.sum(TrF[ellmin:ellmax])**(-0.25)

def battagliaAGN(m200, zs):
    # power law fits:
    rho0  = 4000. * (m200/1e14)**0.29    * (1.+zs)**(-0.66)
    alpha = 0.88  * (m200/1e14)**(-0.03) * (1.+zs)**0.19
    beta  = 3.83  * (m200/1e14)**0.04    * (1.+zs)**(-0.025)
        
    gamma = -0.2
    xc    = 0.5
    return rho0, alpha, beta, gamma, xc

def limber_int(ells,zs,ks,Pzks,hzs,chis):
    hzs = np.array(hzs).reshape(-1)
    chis = np.array(chis).reshape(-1)
    prefactor = hzs / chis**2.

    f = interp2d(ks, zs, Pzks, bounds_error=True)     

    Cells = np.zeros(ells.shape)
    for ii, ell in enumerate(ells):
        kevals = (ell+0.5)/chis

        # hack suggested in https://stackoverflow.com/questions/47087109/evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
        # to get around scipy.interpolate limitations
        interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zs)[0]

        Cells[ii] = np.trapz(interpolated*prefactor, zs)
    return Cells



####### Retired script:

#uell0 = retired_get_uell(zs, rvirs, chis, ellMax)
#Pzell = np.zeros((len(ells), len(zs)))
#f = interp2d(ks, zs, hcos.Pzk, bounds_error=True)     
#for ii, ell in enumerate(ells):
#    kevals = (ell+0.5)/chis
#    interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zs)[0]
#    Pzell[ii, :] = interpolated
#Cl1h = retired_get_Celldtaudtau_1h(zs, ms, nzm, dvols, probell)
#Cl2h = retired_get_Celldtaudtau_2h(zs, ms, nzm, dvols, probell, biases, Pzell)

def retired_get_uell(zs, rvir, chis, ellMax):
    chivir = rvir / aa(zs)[:,None]
    ells = np.arange(ellMax)
    ks = (0.5 + ells[:,None,None]) / chis[None,:,None]
    krv = ks * chivir[None,...]

    num = np.sin(krv) - krv * np.cos(krv)
    return 4.*np.pi * num / ks**3.
def retired_get_Celldtaudtau_1h(zs, ms, nzm, dvol, probell):
    # The 1-halo term
    Cl1hdz = np.trapz(nzm[None,...] * np.abs(probell)**2., ms, axis=2)
    Cl1h   = np.trapz(dvol[None,:] * Cl1hdz, zs, axis=1)
    return Cl1h

def retired_get_Celldtaudtau_2h(zs, ms, nzm, dvol, probell, biases, Pzell):
    # The 2-halo term
    Cl2hdz = np.trapz(nzm[None,...] * biases[None,...] * probell, ms, axis=2)
    Cl2h   = np.trapz(dvol[None,:] * Pzell * Cl2hdz**2., zs, axis=1)
    return Cl2h
