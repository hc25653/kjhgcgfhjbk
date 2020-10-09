import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import struct

#Function that creates a modified rayleigh distribution
def Ray(x,a,b,c,d,exponent):
    return a*(x-b)**2*np.exp(-1*np.power(x-d,2)/(2*c**2))

#Function for generating velocities
def Speed(imin,pols,exponent,arraysize):
    #print(arraysize)
    imin = imin
    imax = 6.
    ndiv = 1001.
    idiv=np.asarray(range(int(ndiv)),dtype=float)*(imax-imin)/(ndiv-1.)+imin
    ibin=np.empty(int(ndiv-1.))
    iavg=np.empty(int(ndiv-1.))
    dA=np.empty(int(ndiv-1.))
    icum=np.empty(int(ndiv-1.))
    
    for i in range(int(ndiv-1)):
        iavg[i]=(idiv[i]+idiv[i+1])/2.
        #print(iavg[i])
        dA[i]=Ray(iavg[i],pols[0],pols[1],pols[2],pols[3],exponent)
        
    totA = np.sum(dA)
    Nparts = dA*arraysize/totA

    #accounting for fractional probability of getting a body at vec
    Nfrac = Nparts - np.floor(Nparts)

    Ntest = np.random.random_sample(int(ndiv)-1)
    sel = np.where(Nfrac>Ntest)[0]
    Nparts[sel] = Nparts[sel]+1
    Nfrac[sel] = -(1.0 - Nfrac[sel])
    Nparts = np.floor(Nparts)

    #making sure number of vec's match number of particles
    arrclip = np.sum(Nparts) - arraysize
    if (arrclip<0):
         for i in range(int(-arrclip)):
             maxadd = np.argmax(Nfrac)
             Nparts[maxadd] = Nparts[maxadd]+1
             Nfrac[maxadd] = 0.0

    if (arrclip>0):
        for i in range(int(arrclip)):
            sel = np.where(Nparts>0)[0]
            mintake = np.argmin(Nfrac[sel])
            Nparts[sel[mintake]] = Nparts[sel[mintake]]-1

    #assigning velocities within each bin
    vel = np.empty(int(np.sum(Nparts)))
    vel = np.array([])
    start = 0
    for i in range(int(ndiv-1)):
        if (Nparts[i] < 0):
            Nparts[i] = Nparts[i] * -1

        vel = np.append(vel,idiv[i]+np.random.random_sample(int(Nparts[i]))*\
                        (idiv[i+1] - idiv[i]))
        start = start+Nparts[i]
        #print(np.append(vel,idiv[i]+\
        #np.random.random_sample(int(Nparts[i]))*(idiv[i+1] - idiv[i])))

    random.shuffle(vel)
    return(vel)

#Mass dispersion without divot
def MassDisp(arraysize,alpha,mx,mn):
    
    #Converting break magnitude into radius
    #Old constant, 11882420.27
    #New constant, 9.16 => 50 km, 16980090.82

    ndiv = 1000
    
    mdiv=np.asarray(range(int(ndiv)),dtype=float)*(mx-mn)/(ndiv-1.)+mn
                
    mavg=np.empty(int(ndiv-1.))
    dA=np.empty(int(ndiv-1.))
    
    #Power 1
    for i in range(int(ndiv-1)):
        mavg[i]=(mdiv[i]+mdiv[i+1])/2.
        dA[i]=10**(alpha*mavg[i])
    
    totA = np.sum(dA)
    Nparts = dA*arraysize/totA
    
        #accounting for fractional probability of getting a body at angle
    Nfrac = Nparts - np.floor(Nparts)

    Ntest = np.random.random_sample(int(ndiv)-1)

    sel = np.where(Nfrac>Ntest)[0]
    Nparts[sel] = Nparts[sel]+1
    Nfrac[sel] = -(1.0 - Nfrac[sel])
    Nparts = np.floor(Nparts)
    
    #making sure number of ang's match number of particles
    arrclip = np.sum(Nparts) - arraysize
    if (arrclip<0):
        for i in range(int(-arrclip)):
            maxadd = np.argmax(Nfrac)
            Nparts[maxadd] = Nparts[maxadd]+1
            Nfrac[maxadd] = 0.0

    if (arrclip>0):
        for i in range(int(arrclip)):
             sel = np.where(Nparts>0)[0]    
             mintake = np.argmin(Nfrac[sel])
             Nparts[sel[mintake]] = Nparts[sel[mintake]]-1
             
    #assigning inclinations within each bin
    mass = np.empty(int(np.sum(Nparts)))
    start = 0
    for i in range(int(ndiv-1)):
        mass[int(start):int(start)+int(Nparts[i])] = mdiv[i]+ \
        np.random.random_sample(int(Nparts[i]))*(mdiv[i+1] - mdiv[i])
        start = start+Nparts[i]
        
    #Converting radius to mass
    #density = 1.3 g/cm3
    
    mass = (1329/np.sqrt(0.04))/2*10**(-0.2*mass)
    
    mass = 500*np.pi*(4.0/3.0)*(mass*1000)**3

    random.shuffle(mass)
    return(mass)

#Mass dispersion with knee
def MassDispDiv(arraysize,bright,faint,mx,mn,brk):
    
    #Converting break magnitude into radius
    
    bright = bright
    faint = faint
    
    #print((1329/np.sqrt(0.04))/2*10**(-0.2*12.6))

    ndiv = 1000
    nbright = np.ceil(float(ndiv)*(brk-mx)/(mn-mx))
    nfaint = float(ndiv)-nbright+1
    
    brightdiv = np.asarray(range(int(nbright)),dtype=float)*1.0*\
                (brk-mx)/(nbright-1.)+mx
    faintdiv = np.asarray(range(int(nfaint)),dtype=float)*1.0*\
                (mn-brk)/(nfaint-1.)+brk
    
                
    brightavg = np.empty(int(nbright-1.))
    faintavg = np.empty(int(nfaint-1.))
    brightdA=np.empty(int(nbright-1.))
    faintdA=np.empty(int(nfaint-1.))

    #Power 1
    for i in range(int(nbright-1)):
        brightavg[i]=(brightdiv[i]+brightdiv[i+1])/2.
        brightdA[i]=10**(bright*brightavg[i])     
        
    #Power 2
    for i in range(int(nfaint-1)):
        faintavg[i]=(faintdiv[i]+faintdiv[i+1])/2.
        faintdA[i]=10**(faint*faintavg[i])*10**(bright*brk)/1225
    
    dA = np.append(brightdA,faintdA)
    brightdiv = brightdiv[:-1]
    idiv = np.append(brightdiv,faintdiv)
    
    totA = np.sum(dA)
    Nparts = dA*arraysize/totA
    
        #accounting for fractional probability of getting a body at angle
    Nfrac = Nparts - np.floor(Nparts)

    Ntest = np.random.random_sample(int(ndiv)-1)

    sel = np.where(Nfrac>Ntest)[0]
    Nparts[sel] = Nparts[sel]+1
    Nfrac[sel] = -(1.0 - Nfrac[sel])
    Nparts = np.floor(Nparts)
    
    #making sure number of ang's match number of particles
    arrclip = np.sum(Nparts) - arraysize
    if (arrclip<0):
        for i in range(int(-arrclip)):
            maxadd = np.argmax(Nfrac)
            Nparts[maxadd] = Nparts[maxadd]+1
            Nfrac[maxadd] = 0.0

    if (arrclip>0):
        for i in range(int(arrclip)):
             sel = np.where(Nparts>0)[0]    
             mintake = np.argmin(Nfrac[sel])
             Nparts[sel[mintake]] = Nparts[sel[mintake]]-1
             
    #assigning inclinations within each bin
    mass = np.empty(int(np.sum(Nparts)))
    start = 0
    for i in range(int(ndiv-1)):
        mass[int(start):int(start)+int(Nparts[i])] = idiv[i]+ \
        np.random.random_sample(int(Nparts[i]))*(idiv[i+1] - idiv[i])
        start = start+Nparts[i]
        
    #Converting radius to mass
    #density = 1.3 g/cm3
    
    mass = (1329/np.sqrt(0.04))/2*10**(-0.2*mass)
    
    mass = 500*np.pi*(4.0/3.0)*(mass*1000)**3
    #mass = mass/2

    random.shuffle(mass)
    return(mass)
    
def Entrance(points):
    Thetas = 2*np.pi*np.random.random_sample(points)
    Phis = np.arccos(2*np.random.random_sample(points) - 1)
    
    return [Thetas,Phis]

def Direction(points):
    #print(points)
    
    Phis = np.arcsin(np.sqrt(np.random.random_sample(points)))
    
    Thetas = np.random.random_sample(points)*np.pi*2
    
    for i in range(len(Phis)):
        if Phis[i] < 2000000/476402000:
            Phis[i] = np.arcsin(np.sqrt(np.random.random_sample(1)))
            i = i - 1
    
    return([Thetas,Phis])
    


###############################################################################
#Toggleable parameters of the encounter file
###############################################################################
#Total Simulation length in years
Total_Time = 4000000000

#Mass of the central object in kg.  For radius calculation
Mass = 2e18

#Collision Rates (Collisions per km squared per year)
#Cold Classical + Cold Classical
Cold_Cold_Rate = 6.33e-22
#Hot Classical + Cold Classical
Hot_Cold_Rate = 2.48e-22
#3:2 Plutinos + Cold Classical
Pluto_Cold_Rate = 2.53e-22

#Calculating Area over which encounters occur (km)
#656 comes from 44au times km/au.  
Hill_Radius = 6.56e9*np.power(Mass/(3*2e30),(1./3.))*8
Hill_Area = 4*np.pi*(Hill_Radius)**2

print(np.power(Mass/(3*2e30),(1./3.))*8*0.07)

#Number of objects in the Kuiper belt as approximated by the code, BodyNumber.py 
Cold_Pop = 214509
Hot_Pop = 104066
Plut_Pop = 38653

#Population numbers are from the body number program
arraysize_C = int(Total_Time*Hill_Area*Cold_Cold_Rate*Cold_Pop)
print("Amount of Cold Interactions: " + str(Total_Time*Hill_Area*Cold_Cold_Rate*Cold_Pop))
arraysize_H = int(Total_Time*Hill_Area*Hot_Cold_Rate*Hot_Pop)
print("Amount of Hot Interactions: " + str(Total_Time*Hill_Area*Hot_Cold_Rate*Hot_Pop))
arraysize_P = int(Total_Time*Hill_Area*Pluto_Cold_Rate*Plut_Pop)
print("Amount of Plutino Interactions: " + str(Total_Time*Hill_Area*Pluto_Cold_Rate*Plut_Pop))

arraysize_C = 30000
arraysize_H = 30000
arraysize_P = 30000

###############################################################################
#Now, We Angles and Masses
###############################################################################

#Generating Masses
Mass_Class = MassDispDiv(arraysize_C,1.2,0.4,4.5,10.2,7.7)
Mass_Hot = MassDispDiv(arraysize_H,0.9,0.4,2.5,10.5,7.7)
Mass_Pluto = MassDispDiv(arraysize_P,0.9,0.4,2.5,10.5,7.7)

#Generating Angles
Angle_Pos_Class = Entrance(arraysize_C)
Angle_Vel_Class = Direction(arraysize_C)

Angle_Pos_Hot = Entrance(arraysize_H)
Angle_Vel_Hot = Direction(arraysize_H)

Angle_Pos_Pluto = Entrance(arraysize_P)
Angle_Vel_Pluto = Direction(arraysize_P)

###############################################################################
#Identifying masses that are too small or far away
###############################################################################

#Identifying irrelevant cold population encounters
Angle_Class_N = []
Mass_Class_List_N = []
arraysize_C_N = arraysize_C

for i in range(0,arraysize_C):
    if Mass_Class[i] < 20.6*10**20:
        if Angle_Vel_Class[1][i] > np.pi/6:
            Mass_Class[i] = -1
            arraysize_C_N -= 1
            
            continue
    
    if Mass_Class[i] < 2.6*10**17:
        if Angle_Vel_Class[1][i] > 0.1253:
            Mass_Class[i] = -1
            arraysize_C_N -= 1
            
            continue
    
#Identifying irrelevant hot population encounters
Angle_Hot_N = []
Mass_Hot_List_N = []
arraysize_H_N = arraysize_H

for i in range(0,arraysize_H):
    if Mass_Hot[i] < 20.6*10**20:
        if Angle_Vel_Hot[1][i] > np.pi/6:
            Mass_Hot[i] = -1
            arraysize_H_N -= 1
            
            continue
    
    if Mass_Hot[i] < 2.6*10**17:
        if Angle_Vel_Hot[1][i] > 0.1253:
            Mass_Hot[i] = -1
            arraysize_H_N -= 1
            
            continue
        
#Identifying irrelevant plutino population encounters
Angle_Pluto_N = []
Mass_Pluto_List_N = []
arraysize_P_N = arraysize_P
for i in range(0,arraysize_P):
    if Mass_Pluto[i] < 2.6*10**17:
        if Angle_Vel_Pluto[1][i] > 0.1253:
            Mass_Pluto[i] = -1
            arraysize_P_N -= 1
            
            continue
    
    if Mass_Pluto[i] < 20.6*10**20:
        if Angle_Vel_Pluto[1][i] > np.pi/6:
            Mass_Pluto[i] = -1
            arraysize_P_N -= 1
            
            continue
    
###############################################################################
#Remaking mass and angle data 
###############################################################################

#Cold Data
Angle_Pos_Class_N = np.zeros((2,arraysize_C_N))
Angle_Vel_Class_N = np.zeros((2,arraysize_C_N))
Mass_Class_N = np.zeros(arraysize_C_N)

j = 0
for i in range(0,arraysize_C):
    if Mass_Class[i] != -1:
        Mass_Class_N[j] = Mass_Class[i]
        Angle_Pos_Class_N[0][j] = Angle_Pos_Class[0][i]
        Angle_Pos_Class_N[1][j] = Angle_Pos_Class[1][i]
        Angle_Vel_Class_N[0][j] = Angle_Vel_Class[0][i]
        Angle_Vel_Class_N[1][j] = Angle_Vel_Class[1][i]
        
        j += 1
        
#Hot Data
Angle_Pos_Hot_N = np.zeros((2,arraysize_H_N))
Angle_Vel_Hot_N = np.zeros((2,arraysize_H_N))
Mass_Hot_N = np.zeros(arraysize_H_N)

j = 0
for i in range(0,arraysize_H):
    if Mass_Hot[i] != -1:
        Mass_Hot_N[j] = Mass_Hot[i]
        Angle_Pos_Hot_N[0][j] = Angle_Pos_Hot[0][i]
        Angle_Pos_Hot_N[1][j] = Angle_Pos_Hot[1][i]
        Angle_Vel_Hot_N[0][j] = Angle_Vel_Hot[0][i]
        Angle_Vel_Hot_N[1][j] = Angle_Vel_Hot[1][i]
        
        j += 1
        
#Pluto Data
Angle_Pos_Pluto_N = np.zeros((2,arraysize_P_N))
Angle_Vel_Pluto_N = np.zeros((2,arraysize_P_N))
Mass_Pluto_N = np.zeros(arraysize_P_N)
        
j = 0
for i in range(0,arraysize_P):
    if Mass_Pluto[i] != -1:
        Mass_Pluto_N[j] = Mass_Pluto[i]
        Angle_Pos_Pluto_N[0][j] = Angle_Pos_Pluto[0][i]
        Angle_Pos_Pluto_N[1][j] = Angle_Pos_Pluto[1][i]
        Angle_Vel_Pluto_N[0][j] = Angle_Vel_Pluto[0][i]
        Angle_Vel_Pluto_N[1][j] = Angle_Vel_Pluto[1][i]
        
        j += 1
        
###############################################################################
#Reading velocity data from files
###############################################################################
        
#Taking data from simulation files to make approximate velocity data
    
#directory where this is taking place
start = ""

#Making place for data to be imported to
Data_Fine_C = []
Data_Fine_H = []
Data_Fine_P = []

#Making it easier to pull specific information about an encounter
class Encounter:
    def __init__(self,number,AU1,AU2,Ec1,Ec2,In1,In2,Delta,Smin):

        self.number = number
        self.AU1 = AU1
        self.AU2 = AU2
        self.Ec1 = Ec1
        self.Ec2 = Ec2
        self.In1 = In1
        self.In2 = In2
        self.Delta = Delta*1731.46
        self.Smin = Smin

#Importing data from all directories
for i in range(1,301):
    Data_Fine_C.append(np.genfromtxt(start+"Class+Class/encounterlist_fine"+str(i)+'.txt'))
    Data_Fine_H.append(np.genfromtxt(start+"Hot+Class/encounterlist_fine"+str(i)+'.txt'))
    Data_Fine_P.append(np.genfromtxt(start+"Pluto+Class/encounterlist_fine"+str(i)+'.txt'))

#Orginizing Fine encounter data into encounter class    
Fine_Encounters_Class = []
Fine_Encounters_Hot = []
Fine_Encounters_Pluto = []

for j in range(0,len(Data_Fine_C)):
    
    for i in range(1,len(Data_Fine_C[j])):     
        
        Fine_Encounters_Class.append(Encounter(i-1,Data_Fine_C[j][i,0],\
                Data_Fine_C[j][i,3],Data_Fine_C[j][i,1],Data_Fine_C[j][i,4],\
                Data_Fine_C[j][i,2],Data_Fine_C[j][i,5],Data_Fine_C[j][i,6],\
                Data_Fine_C[j][i,7]))
        
for j in range(0,len(Data_Fine_H)):
    for i in range(1,len(Data_Fine_H[j])):        
        
        Fine_Encounters_Hot.append(Encounter(i-1,Data_Fine_H[j][i,0],\
                Data_Fine_H[j][i,3],Data_Fine_H[j][i,1],Data_Fine_H[j][i,4],\
                Data_Fine_H[j][i,2],Data_Fine_H[j][i,5],Data_Fine_H[j][i,6],\
                Data_Fine_H[j][i,7]))
        
for j in range(0,len(Data_Fine_P)):
    for i in range(1,len(Data_Fine_P[j])):        
        
        Fine_Encounters_Pluto.append(Encounter(i-1,Data_Fine_P[j][i,0],\
                Data_Fine_P[j][i,3],Data_Fine_P[j][i,1],Data_Fine_P[j][i,4],\
                Data_Fine_P[j][i,2],Data_Fine_P[j][i,5],Data_Fine_P[j][i,6],\
                Data_Fine_P[j][i,7]))

#Taking out specific velocity information
Hist_Data_Fine_Class = []
for i in range(0,len(Fine_Encounters_Class)):
    Hist_Data_Fine_Class.append(Fine_Encounters_Class[i].Delta)
Hist_Data_Fine_Hot = []
for i in range(0,len(Fine_Encounters_Hot)):
    Hist_Data_Fine_Hot.append(Fine_Encounters_Hot[i].Delta)
Hist_Data_Fine_Pluto = []
for i in range(0,len(Fine_Encounters_Pluto)):
    Hist_Data_Fine_Pluto.append(Fine_Encounters_Pluto[i].Delta)      

###############################################################################
#Fitting functions to data and generating velocities from them
###############################################################################

#Each data type fits to a different function so this will be done seperatly
    
#A modified Rayleigh function works best
def Ray3(x,a,b,c,d):
    return a*(x-b)**3*np.exp(-1*np.power(x-d,2)/(2*c**2))
def Ray2(x,a,b,c,d):
    return a*(x-b)**2*np.exp(-1*np.power(x-d,2)/(2*c**2))


#Starting with the classical + classical   

bins = plt.hist(Hist_Data_Fine_Class, bins=100)

total = len(Hist_Data_Fine_Class)
bin_probs = []
for i in range(0,len(bins[0])):
    bin_probs.append(bins[0][i]/total)

popt, pcov = curve_fit(Ray3, bins[1], np.append(bin_probs,0),\
                       p0 = [0.021,0,1,-4])

Speeds_Class = Speed(min(bins[1]),popt,3,arraysize_C_N)*0.000577548

#Continuing with the Hot + classical

bins = plt.hist(Hist_Data_Fine_Hot, bins=100)

total = len(Hist_Data_Fine_Hot)
bin_probs = []
for i in range(0,len(bins[0])):
    bin_probs.append(bins[0][i]/total)

popt, pcov = curve_fit(Ray2, bins[1], np.append(bin_probs,0),\
                       p0 = [0.021,0,1,-4])

Speeds_Hot = Speed(min(bins[1]),popt,2,arraysize_H_N)*0.000577548

#Ending with the Pluto + classical

bins = plt.hist(Hist_Data_Fine_Pluto, bins=100)

total = len(Hist_Data_Fine_Pluto)
bin_probs = []
for i in range(0,len(bins[0])):
    bin_probs.append(bins[0][i]/total)

popt, pcov = curve_fit(Ray2, bins[1], np.append(bin_probs,0),\
                       p0 = [0.021,0,1,-4])

#Converting km/s to au/day
Speeds_Pluto = Speed(min(bins[1]),popt,2,arraysize_P_N)*0.000577548
plt.show()

#Getting rid of anything that is too slow, these break SWIFT
for i in range(0,len(Speeds_Class)):
    if Speeds_Class[i] < 0.02*0.000577548:
        Speeds_Class[i] = 0.02*0.000577548
    #print(Speeds[i])

for i in range(0,len(Speeds_Hot)):
    if Speeds_Hot[i] < 0.02*0.000577548:
        Speeds_Hot[i] = 0.02*0.000577548

for i in range(0,len(Speeds_Pluto)):
    if Speeds_Pluto[i] < 0.02*0.000577548:
        Speeds_Pluto[i] = 0.02*0.000577548

############################################################################### 
#Now we generate the inidial positions of the passing bodies
############################################################################### 
    
#Generating Initial Entrance For Cold Bodies

xClass = Hill_Radius*np.sin(Angle_Pos_Class_N[1])*np.cos(Angle_Pos_Class_N[0])*6.68459e-9
yClass = Hill_Radius*np.sin(Angle_Pos_Class_N[1])*np.sin(Angle_Pos_Class_N[0])*6.68459e-9
zClass = Hill_Radius*np.cos(Angle_Pos_Class_N[1])*6.68459e-9

#Generating Initial Entrance For Hot Bodies
xHot = Hill_Radius*np.sin(Angle_Pos_Hot[1])*np.cos(Angle_Pos_Hot[0])*6.68459e-9
yHot = Hill_Radius*np.sin(Angle_Pos_Hot[1])*np.sin(Angle_Pos_Hot[0])*6.68459e-9
zHot = Hill_Radius*np.cos(Angle_Pos_Hot[1])*6.68459e-9

#Generating Initial Entrance For Pluto Bodies
xPluto = Hill_Radius*np.sin(Angle_Pos_Pluto[1])*np.cos(Angle_Pos_Pluto[0])*6.68459e-9
yPluto = Hill_Radius*np.sin(Angle_Pos_Pluto[1])*np.sin(Angle_Pos_Pluto[0])*6.68459e-9
zPluto = Hill_Radius*np.cos(Angle_Pos_Pluto[1])*6.68459e-9

###############################################################################
#Now, We Calculate the Initial Velocity
###############################################################################


#Generating Velocities.  Doing this in three parts to save me mental energy

#Affect from the initial direction
###############################################################################

dxClass = Speeds_Class*np.sin(Angle_Vel_Class_N[1])*np.cos(Angle_Vel_Class_N[0])
dyClass = Speeds_Class*np.sin(Angle_Vel_Class_N[0])*np.sin(Angle_Vel_Class_N[1])
dzClass = -1*Speeds_Class*np.cos(Angle_Vel_Class_N[1])

###############################################################################

dxHot = Speeds_Hot*np.sin(Angle_Vel_Hot_N[1])*np.cos(Angle_Vel_Hot_N[0])
dyHot = Speeds_Hot*np.sin(Angle_Vel_Hot_N[0])*np.sin(Angle_Vel_Hot_N[1])
dzHot = -1*Speeds_Hot*np.cos(Angle_Vel_Hot_N[1])

###############################################################################

dxPluto = Speeds_Pluto*np.sin(Angle_Vel_Pluto_N[1])*np.cos(Angle_Vel_Pluto_N[0])
dyPluto = Speeds_Pluto*np.sin(Angle_Vel_Pluto_N[0])*np.sin(Angle_Vel_Pluto_N[1])
dzPluto = -1*Speeds_Pluto*np.cos(Angle_Vel_Pluto_N[1])

#Rotation 1
#Due to me being an idiot, i made this code incorrectly and the best fix is to
#use the preserv.  This is because after I change dx, I have to use the 
#original dx to change dz.  There's probably a better way to do this but I 
#couldn't think of it in time.
###############################################################################

preserv = dxClass
dxClass = dxClass*np.cos(Angle_Pos_Class_N[1])+dzClass*\
    np.sin(Angle_Pos_Class_N[1])
dyClass = dyClass
dzClass = -1*preserv*np.sin(Angle_Pos_Class_N[1])+dzClass*\
    np.cos(Angle_Pos_Class_N[1])

###############################################################################
preserv = dxHot
dxHot = dxHot*np.cos(Angle_Pos_Hot_N[1])+dzHot*\
    np.sin(Angle_Pos_Hot_N[1])
dyHot = dyHot
dzHot = -1*preserv*np.sin(Angle_Pos_Hot_N[1])+dzHot*\
    np.cos(Angle_Pos_Hot_N[1])

###############################################################################
preserv = dxPluto
dxPluto = dxPluto*np.cos(Angle_Pos_Pluto_N[1])+dzPluto*\
    np.sin(Angle_Pos_Pluto_N[1])
dyPluto = dyPluto
dzPluto = -1*preserv*np.sin(Angle_Pos_Pluto_N[1])+dzPluto*\
    np.cos(Angle_Pos_Pluto_N[1])

#Rotation 2
#Same preserv thing as before
###############################################################################
preserv = dxClass
dxClass = (dxClass*np.cos(Angle_Pos_Class_N[0])-dyClass*\
    np.sin(Angle_Pos_Class_N[0]))
dyClass = (preserv*np.sin(Angle_Pos_Class_N[0])+dyClass*\
    np.cos(Angle_Pos_Class_N[0]))
dzClass = dzClass

###############################################################################
preserv = dxHot
dxHot = (dxHot*np.cos(Angle_Pos_Hot_N[0])-dyHot*\
    np.sin(Angle_Pos_Hot_N[0]))
dyHot = (preserv*np.sin(Angle_Pos_Hot_N[0])+dyHot*\
    np.cos(Angle_Pos_Hot_N[0]))
dzHot = dzHot

###############################################################################
preserv = dxPluto
dxPluto = (dxPluto*np.cos(Angle_Pos_Pluto_N[0])-dyPluto*\
    np.sin(Angle_Pos_Pluto_N[0]))
dyPluto = (preserv*np.sin(Angle_Pos_Pluto_N[0])+dyPluto*\
    np.cos(Angle_Pos_Pluto_N[0]))
dzPluto = dzPluto

###############################################################################
#Now lets calculate the rough impulse of the interaction
###############################################################################

#Impulse in this case is impact parameter divided by velocity
#Units are days, last number is to convert hill radius into AU

imp_Class = np.sin(Angle_Vel_Class_N[1])*Hill_Radius/Speeds_Class*6.68459e-9
imp_Hot = np.sin(Angle_Vel_Hot_N[1])*Hill_Radius/Speeds_Hot*6.68459e-9
imp_Pluto = np.sin(Angle_Vel_Pluto_N[1])*Hill_Radius/Speeds_Pluto*6.68459e-9

print('Cold Survivor Chance ' + str(arraysize_C_N/arraysize_C*100) + "%")
print('Hot Survivor Chance ' + str(arraysize_H_N/arraysize_H*100) + "%")
print('Pluto Survivor Chance ' + str(arraysize_P_N/arraysize_P*100) + "%")

###############################################################################
#Now, We write the file for the classical self interaction
###############################################################################
#First line is the number of interactions
#The next lines are data fro each body that next interacts

#rough approximation of impulse (days)
#time that intaraction occurs (years)
#mass of interactor (Msun)
#Spacial Position that interactor enters from (AU)
#Velocity Components of interactor (AU/day)

###############################################################################
#While I am making 3 seperate encounter files for each group, i am making a big
#Master file of all encounters
Sakhr_All = open("Kul_Sakhr", "wb")
Sakhr_All.write(bytearray(struct.pack("h",4)))
Sakhr_All.write(bytearray(struct.pack("h",0)))
Sakhr_All.write(bytearray(struct.pack("i",arraysize_C_N+\
    arraysize_H_N+arraysize_P_N)))
Sakhr_All.write(bytearray(struct.pack("h",4)))
Sakhr_All.write(bytearray(struct.pack("h",0)))


###############################################################################
#Now, the process of putting everything in the file in order of time
###############################################################################

T_Class = np.linspace(0,1,arraysize_C_N)*Total_Time*365.26
T_Hot = np.linspace(0,1,arraysize_H_N)*Total_Time*365.26
T_Pluto = np.linspace(0,1,arraysize_P_N)*Total_Time*365.26

T_Class = np.append(T_Class,Total_Time*365.26 + 1)
T_Hot = np.append(T_Hot,Total_Time*365.26 + 1)
T_Pluto = np.append(T_Pluto,Total_Time*365.26 + 1)

j_C = 0
j_H = 0
j_P = 0

for i in range(0,arraysize_C_N + arraysize_H_N + arraysize_P_N):
    
    choice = [T_Class[j_C],T_Hot[j_H],T_Pluto[j_P]]
    choice.sort()
    
    if T_Class[j_C] == choice[0]:
        
        Mass = 1.48775255e-34*Mass_Class_N[j_C]
        
        Sakhr_All.write(bytearray(struct.pack("h",36)))
        Sakhr_All.write(bytearray(struct.pack("h",0)))
        Sakhr_All.write(bytearray(struct.pack("f",imp_Class[j_C])))
        Sakhr_All.write(bytearray(struct.pack("f",T_Class[j_C])))
        Sakhr_All.write(bytearray(struct.pack("f",Mass)))
        Sakhr_All.write(bytearray(struct.pack("f",xClass[j_C])))
        Sakhr_All.write(bytearray(struct.pack("f",yClass[j_C])))
        Sakhr_All.write(bytearray(struct.pack("f",zClass[j_C])))
        Sakhr_All.write(bytearray(struct.pack("f",dxClass[j_C])))
        Sakhr_All.write(bytearray(struct.pack("f",dyClass[j_C])))
        Sakhr_All.write(bytearray(struct.pack("f",dzClass[j_C])))
        Sakhr_All.write(bytearray(struct.pack("h",36)))
        Sakhr_All.write(bytearray(struct.pack("h",0)))
        
        j_C += 1
        
    elif T_Hot[j_H] == choice[0]:
        
        Mass = 1.48775255e-34*Mass_Hot_N[j_H]
        
        Sakhr_All.write(bytearray(struct.pack("h",36)))
        Sakhr_All.write(bytearray(struct.pack("h",0)))
        Sakhr_All.write(bytearray(struct.pack("f",imp_Hot[j_H])))
        Sakhr_All.write(bytearray(struct.pack("f",T_Hot[j_H])))
        Sakhr_All.write(bytearray(struct.pack("f",Mass)))
        Sakhr_All.write(bytearray(struct.pack("f",xHot[j_H])))
        Sakhr_All.write(bytearray(struct.pack("f",yHot[j_H])))
        Sakhr_All.write(bytearray(struct.pack("f",zHot[j_H])))
        Sakhr_All.write(bytearray(struct.pack("f",dxHot[j_H])))
        Sakhr_All.write(bytearray(struct.pack("f",dyHot[j_H])))
        Sakhr_All.write(bytearray(struct.pack("f",dzHot[j_H])))
        Sakhr_All.write(bytearray(struct.pack("h",36)))
        Sakhr_All.write(bytearray(struct.pack("h",0)))
        
        j_H += 1
        
    else:
        Mass = 1.48775255e-34*Mass_Pluto_N[j_P]
        
        Sakhr_All.write(bytearray(struct.pack("h",36)))
        Sakhr_All.write(bytearray(struct.pack("h",0)))
        Sakhr_All.write(bytearray(struct.pack("f",imp_Pluto[j_P])))
        Sakhr_All.write(bytearray(struct.pack("f",T_Pluto[j_P])))
        Sakhr_All.write(bytearray(struct.pack("f",Mass)))
        Sakhr_All.write(bytearray(struct.pack("f",xPluto[j_P])))
        Sakhr_All.write(bytearray(struct.pack("f",yPluto[j_P])))
        Sakhr_All.write(bytearray(struct.pack("f",zPluto[j_P])))
        Sakhr_All.write(bytearray(struct.pack("f",dxPluto[j_P])))
        Sakhr_All.write(bytearray(struct.pack("f",dyPluto[j_P])))
        Sakhr_All.write(bytearray(struct.pack("f",dzPluto[j_P])))
        Sakhr_All.write(bytearray(struct.pack("h",36)))
        Sakhr_All.write(bytearray(struct.pack("h",0)))
        
        j_P += 1

Sakhr_All.close()