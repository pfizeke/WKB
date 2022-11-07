# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:31:26 2019

@author: Ivan
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte
import scipy.special as spec
import scipy.optimize as opt
import scipy.interpolate as inter

#Fundamental constants
h=6.62607004e-34 #Planck's constant, J*s
amu=1.6605e-27 #atomic mass unit in kg
c=29979245800 #speed of light in cm/s

#Molecular parameters
m=27.97693*15.99492/43.97185 #Reduced mass
data=np.loadtxt('X_state.txt') #Potential energy curve (PEC)
xp=data[:,0]
fp=data[:,1]
V=inter.interp1d(xp,fp,'cubic') #Interpolated PEC

#Spectroscopic constants of SiO+
we=1153.2 #Vibration freqiency, cm-1
wexe=6.5034 #1st anharmonicity
weye=-4.8954E-03 #2nd anharmonicity
Be=7.1147E-01 #Rotational constant
De=1.0973E-06 #1st centrifugal distortion
He=-1.4534E-13 #2nd centrifugal distortion
Ke=-4.3723E-18 #3rd centrifugal distortion

#SiO+ rotational and vibrational quantum state
NRot=0 #Rotational quantum number
N2=NRot*(NRot+1)
vib=1 #Vibrational level
Bn=Be-De*N2+He*N2**2+Ke*N2**3 #Rotational constant with centrifugal distortion corrections
NRG=N2*Bn+we*(vib+0.5)-wexe*(vib+0.5)**2+weye*(vib+0.5)**3 #Energy expression

def CV(r,L): #Centrifugally distorted PEC
    return h/c/8/np.pi**2/m/amu/r**2*1e20*L*(L+1)+V(r)


x=np.linspace(1.2,5.0,1000) #grid for plotting potential energy curve
plt.plot(x,CV(x,NRot),label='N = '+str(NRot)) #Plot PEC w/centrufugal correction
plt.xlabel('R,$\AA$')
plt.ylabel('E,cm$^{-1}$')
 
#Defining wavefunctions
def p(r,E,L): #Momentum of SiO+ vibration
    return 1e-10*np.sqrt(8*m*amu/h*c*abs(E-CV(r,L)))
def wfb(r,E,L,tpi): #Classically allowed wavefunction in the bound region
    phase=inte.quad(p,tpi,r,(E,L))
    return 2*np.sin(np.pi*(phase[0]+0.25))/np.sqrt(p(r,E,L)) #phase[0]
def wfi(r,E,L,tpi): #Classically forbidden tunneling wavefunction before the inner turning point
    phase=inte.quad(p,r,tpi,(E,L))
    return np.exp(-np.pi*phase[0])/np.sqrt(p(r,E,L))
def wfo(r,E,L,tpo): #Classically forbidden tunneling wavefunction after the outer turning point
    phase=inte.quad(p,tpo,r,(E,L))
    return np.exp(-np.pi*phase[0])/np.sqrt(p(r,E,L))

#Finding turning points
def TP(r,E,L): #Kinetic energy of SiO+ vibration for finding turning points
    return E-CV(r,L)
def shoot(E,L,phase0): #WKB quantization function
    r1=opt.least_squares(TP,1.3,args=(E,L))
    r2=opt.least_squares(TP,2.0,args=(E,L))
    return inte.quad(p,r1.x,r2.x,(E,L))[0]-phase0

NRG=opt.least_squares(shoot,NRG,args=(NRot,vib+0.5)).x[0] #Finding energy eigenvalue via WKB quantization
res1=opt.least_squares(TP,1.3,args=(NRG,NRot)) #Inner turning point
res2=opt.least_squares(TP,2.0,args=(NRG,NRot)) #Outer turning point
plt.plot([res1.x,res2.x],[NRG,NRG]) #Plotting energy level of SiO+ vibration

#Interpolating WKB wavefunctions
step=10*(vib+1) #Grid size for each region
y=np.linspace(res1.x+0.04,res2.x-0.04,step) #Grid for the bound region
yi=np.linspace(1.25,res1.x-0.03,step) #Grid before the inner turning point
yo=np.linspace(res2.x+0.03,2.5,step) #Grid after the outer turning point
z=np.zeros(step) #Wavefunction array for the bound region
zi=np.zeros(step) #Wavefunction array before the inner turning point
zo=np.zeros(step) #Wavefunction array after the outer turning point
for i in range(step): #Populating wavefunction arrays w/values
    z[i]=wfb(y[i],NRG,NRot,res1.x)
    zi[i]=wfi(yi[i],NRG,NRot,res1.x)
    if vib%2==0:
        zo[i]=wfo(yo[i],NRG,NRot,res2.x)
    else:
        zo[i]=-wfo(yo[i],NRG,NRot,res2.x)
ytot=np.append(yi,np.append(y,yo)) #Stitching grid regions
ztot=np.append(zi,np.append(z,zo)) #Stitchina wavefunction regions
wfx0=inter.interp1d(ytot,ztot,'cubic') #Interpolating wavefunction
arg=np.linspace(1.25,2.5,250) #Grid for normalization and plotting wavefunction
normx=np.sqrt(inte.simps(wfx0(arg)**2,arg)) #Normalizing wavefunction
plt.plot(arg,150*wfx0(arg)**2/normx+NRG) #Plotting wavefunction

print('Inner turning point is',round(res1.x[0],3),'Angstrom')
print('Outer turning point is',round(res2.x[0],3),'Angstrom')
bound=inte.quad(p,res1.x,res2.x,(NRG,NRot))

print('At energy =',NRG,'and N =',NRot)
print('bound phase',bound[0])

