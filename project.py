# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 22:23:10 2019

@author: Harish Reddy
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
from math import log, sqrt, exp
from scipy.optimize import least_squares
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

########################################## question 1*######################################


####################################Q1################################################

OIS=pd.read_excel('IR Data.xlsx')
OIS=OIS.iloc[:,0:3]
print(OIS)

tenor=[0.5,1,2,3,4,5,7,10,15,20,30]

discount_factor=[]
for i in range(11):
    D=1/(1+OIS.iloc[i,2]/360)**(tenor[i]*360)
    discount_factor.append(D)

print("The OIS discount factor is",discount_factor)

f1=interp1d(tenor,discount_factor) #OIS interpolation
xnew1=np.linspace(0.5,30) #different tenor

OIS_D=[]
for i in np.arange(0.5,30.5,0.5):
    OIS_D.append(f1(i).tolist())
    

print("---------------------------------------------------------------")
fig1=plt.figure()
ax1=fig1.subplots()
ax1.plot(xnew1,f1(xnew1),tenor,discount_factor,'o')
ax1.set_title("OIS Discount Factor")
ax1.set_xlabel("Time")
ax1.set_ylabel("Discount Factor")

###################################Q2##################################################

IRS=pd.read_excel('IR Data.xlsx')
IRS=IRS.iloc[:,0:3]
print(IRS)
#IRS['Tenor2']=[0.5,1,2,3,4,5,7,10,15,20,30]
IRS_rate=IRS['Rate'].tolist()
tenor=[0.5,1,2,3,4,5,7,10,15,20,30]

PV_fix=[0]*11
PV_flt=[0]*11
Libor_D=[0]*11
Libor_D[0]=1/(1+IRS_rate[0]*tenor[0])
PV_fix=[IRS_rate[i]*sum(OIS_D[s] for s in range(int(tenor[i]*2))) for i in range(11)]

def PV_IRS(i,ds):
    S=[]
    for s in range(int(tenor[i-1]*2),int(tenor[i]*2)):
        S.append(2*OIS_D[s]*((Libor_D[i-1]-ds)/(2*(tenor[i]-tenor[i-1])))/(Libor_D[i-1]
        -(s-int(tenor[i-1])*2+1)*(Libor_D[i-1]-ds)/((2*(tenor[i]-tenor[i-1])))))
    
    PV_flt[i]=PV_fix[i-1]+sum(S)
    return float(PV_fix[i]-PV_flt[i])
        
for i in range(1,11):
    ss=fsolve(lambda x:PV_IRS(i,x),0.5)[0]
    Libor_D[i]=ss
print(Libor_D)

x=[int((tenor[i+1]-tenor[i])*2) for i in range(10)]

f=[Libor_D[i]-Libor_D[i+1] for i in range(10)]
LD=[Libor_D[0]]
for i in range(1,10):
    for s in range(x[i]):
        LD.append(Libor_D[i]-f[i]*s/x[i])
LD.append(Libor_D[-1])
#V libor discount factor per 0.5y
fig2=plt.figure()
axe2=fig2.subplots()
axe2.plot(np.arange(0.5,30.5,0.5),LD)
axe2.set_title('Libor Discount Factor')

################################Q3###############################################
rate2=np.arange(1,30.5,0.5)
f2=interp1d(rate2,LD[1:]) #libor discount factor interpolation

'''1y'''
rate1=np.arange(1.5,10.5,0.5) #we start from 1.5y instead of 1y
D=[] #libor discount factor from 1.5 to 10y
for i in rate1:
    D.append(f2(i).tolist())

swap_1y=[]
for i in range(1,10+1):
    rate_1y=(f2(1)-f2(i+1))/(0.5*sum(D[0:i*2]))
    swap_1y.append(rate_1y)

f1y=interp1d(range(1,11),swap_1y) #1y swap interpolation

print("---------------------------------------------------------------")
swap_1y1y=f1y(1).tolist()
print("1y x 1y",swap_1y1y)
swap_1y2y=f1y(2).tolist()
print("1y x 2y",swap_1y2y)
swap_1y3y=f1y(3).tolist()
print("1y x 3y",swap_1y3y)
swap_1y5y=f1y(5).tolist()
print("1y x 5y",swap_1y5y)
swap_1y10y=f1y(10).tolist()
print("1y x 10y",swap_1y10y)

'''5y'''
rate1=np.arange(5.5,15.5,0.5) #we start from 5.5y instead of 5y
D=[] #libor discount factor from 5 to 15y
for i in rate1:
    D.append(f2(i).tolist())

swap_5y=[]
for i in range(1,10+1):
    rate_5y=(f2(5)-f2(i+5))/(0.5*sum(D[0:i*2]))
    swap_5y.append(rate_5y)

f5y=interp1d(range(1,11),swap_5y) #1y swap interpolation

print("---------------------------------------------------------------")
swap_5y1y=f5y(1).tolist()
print("5y x 1y",swap_5y1y)
swap_5y2y=f5y(2).tolist()
print("5y x 2y",swap_5y2y)
swap_5y3y=f5y(3).tolist()
print("5y x 3y",swap_5y3y)
swap_5y5y=f5y(5).tolist()
print("5y x 5y",swap_5y5y)
swap_5y10y=f5y(10).tolist()
print("5y x 10y",swap_5y10y)

'''10y'''
rate1=np.arange(10.5,20.5,0.5) #we start from 5.5y instead of 5y
D=[] #libor discount factor from 5 to 15y
for i in rate1:
    D.append(f2(i).tolist())

swap_10y=[]
for i in range(1,10+1):
    rate_10y=(f2(10)-f2(i+10))/(0.5*sum(D[0:i*2]))
    swap_10y.append(rate_10y)

f10y=interp1d(range(1,11),swap_10y) #1y swap interpolation

print("---------------------------------------------------------------")
swap_10y1y=f10y(1).tolist()
print("10y x 1y",swap_10y1y)
swap_10y2y=f10y(2).tolist()
print("10y x 2y",swap_10y2y)
swap_10y3y=f10y(3).tolist()
print("10y x 3y",swap_10y3y)
swap_10y5y=f10y(5).tolist()
print("10y x 5y",swap_10y5y)
swap_10y10y=f10y(10).tolist()
print("10y x 10y",swap_10y10y)

p = [1,2,3,5,10]
    
list_1 = [f1y(i).tolist() for i in p]
list_2 = [f5y(i).tolist() for i in p]
list_3 = [f10y(i).tolist() for i in p]
    
forward = list_1 + list_2 + list_3

##################################  question 2   #################################################################

df = pd.read_excel('IR Data.xlsx', 'Swaption',header=2)


alpha_values =[]

rho_values =[]

nu_values = []

optimum =[]

for i in range(len(df)):
    
    input_row_number = i+1
    
    row = input_row_number - 1  # input the row you want to compute in the 'df' dataframe
    
    T = int((df['Expiry'][row])[:-1]) #int((df['Expiry'][row])[:-1] # this is time you start your swap ( time to expiry)
    
    start = (1 + 2*(T-1)) + 1
    
    end  =  start + 2* int((df['Tenor'][row][:-1]))
    
    P = 0.5*(sum(LD[ start : end ]))
    
    S=forward[i]*100
     
  
    
    def Black76LognormalCall(S, K, sigma,T):
        d1 = (log(S/K)+(sigma**2/2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        return (S*norm.cdf(d1) - K*norm.cdf(d2))
    
    def Black_put(S,K,sigma,T):
        d1 = (log(S/K)+(sigma**2/2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        return ( K*norm.cdf(-d2) - S*norm.cdf(-d1) )
    
    def BachelierCall(S, K, sigma, T):
        d = (S-K)/(S*sigma*np.sqrt(T))
        return ( (S-K)*norm.cdf(d) + (S*sigma*np.sqrt(T)*norm.pdf(d)) )
    
    def Displaced_diffusion(S,K,sigma,T,beta):
        
        price = Black76LognormalCall(S/beta, (K+((1-beta)*S/beta)), sigma*beta, T)
        
        return price
        
    def Displaced_diffusion_put(S,K,sigma,T,beta):
        
        price = Black_put(S/beta, (K+((1-beta)*S/beta)), sigma*beta, T)
        
        return price
    
    
    def Implied_Vol_Displaced(S,K,T,beta,sigma):
        
        price = P*Displaced_diffusion(S,K,atm_sigma,T,beta)
        
        H = lambda x : P*Black76LognormalCall(S, K, x,T)-price
        
        Volatility = brentq(H,-0.0001,1)
        
        return (Volatility)
    
    def Implied_Vol_Normal(S,K,T,sigma):
        
        price =  P*BachelierCall(S, K, atm_sigma,T)
        
        H = lambda x :  P*Black76LognormalCall(S, K, x,T)-price
        
        Volatility =  brentq(H,-0.0001,1)
        
        return (Volatility)
        
    
    h = [-200,-150,-100,-50,-25,0,25,50,100,150,200]
    
    
    strikes = [( int(x)/100 + S) for x in h]    # this will give strikes in percentage
    
    market_IV = df.loc[row,'-200bps':].tolist()   #  row here is the input you have to give - row = 0 means it is first row in 'df' dataframe
    
    #atm_sigma = df.loc[row,'ATM']/100   

    atm_sigma = df['ATM'][row]/100
    
    
    ################# optimumm beta
    
    be = np.arange(0.1,1,0.01)
    var = []     
    dff = pd.DataFrame({'strikes':strikes})
    
    dff['market']= market_IV
    
    for h in be:
        beta=h
        c='beta='+str(np.round(beta,1))
        
        for j in range(len(dff)):  
            
            K = dff.loc[j,'strikes']
            
            #sigma = (df.iloc[i,j+2])/100
            
            #sigma = atm_sigma
            
            dff.loc[j,c] =  Implied_Vol_Displaced(S,K,T,beta,atm_sigma)*100
     
    l = dff.columns.tolist()
    
    xx = []
    
    for p in range(len(l)-2):
        xx.append(((dff.iloc[:,1]-dff.iloc[:,p+2])**2).mean())
        
    index=np.argmin(xx)
    
    optimum_beta =  np.round(be[index],2)
    
    optimum.append(optimum_beta)
    
################################################################
    
    betas = [0.2,0.4,0.6,0.8,1]
    
    
    col_names = ['strike','beta_0.2','beta_0.4','beta_0.6','beta_0.8','beta_1']
    
    Df = pd.DataFrame(columns=col_names)
    
    Df['strike'] = strikes
    
        
    for beta in betas:
        
        c = 'beta_' + str(np.round(beta,1))
              
       
        for r in range(len(Df)):
            
            K = Df.loc[r,'strike']
            
            #sigma = (df.iloc[i,j+2])/100
            
            #sigma = atm_sigma
            
            Df.loc[r,c] =  Implied_Vol_Displaced(S,K,T,beta,atm_sigma)*100
            
                                
    for o in range(len(Df)):
        
        K = Df.loc[o,'strike']
        
        #sigma = df.iloc[i,j+2]/100
        
        #sigma = atm_sigma
        
        Df.loc[o,'Normal implied Vol'] = (Implied_Vol_Normal(S,K,T,atm_sigma))*100
                        
       
    Df['Market_IV'] = market_IV
    
        
    def SABR(F, K, T, alpha, beta, rho, nu):
        X = K
        if F == K:
            numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
            numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
            numer3 = ((2 - 3*rho*rho)/24)*nu*nu
            VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
            sabrsigma = VolAtm
        else:
            z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
            zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
            numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
            numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
            numer3 = ((2 - 3*rho*rho)/24)*nu*nu
            numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
            denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
            denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
            denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
            sabrsigma = numer/denom
    
        return sabrsigma
    
    def sabrcalibration(x, strikes, vols, F, T):
        err = 0.0
        for i, vol in enumerate(vols):
            err += (vol - SABR(F, strikes[i], T,
                               x[0], 0.8, x[1], x[2]))**2
    
        return err
    
    
    initialGuess = [0.02, 0.2, 0.1]
    
    Df['Market_IV_indecimal'] = Df['Market_IV']/100
    
    res = least_squares(lambda x: sabrcalibration(x,
                                                  Df['strike'],
                                                  Df['Market_IV_indecimal'],
                                                  S,
                                                  T),
                        initialGuess)
    
    alpha = res.x[0]
    beta = 0.8
    rho = res.x[1]
    nu = res.x[2]
    
    alpha_values.append(alpha)
    
    rho_values.append(rho)
    
    nu_values.append(nu)
    
    
    
    for i in range(len(Df)):
        Df.loc[i, 'SABR IV'] = SABR(S, Df.loc[i, 'strike'], T, alpha, beta, rho, nu)*100
    
    
    fig3 = plt.figure(figsize=(14,10))
    
    ax3 = fig3.add_subplot(111)
    
    ax3.plot(strikes,Df['beta_0.4'],linewidth=3.0)  
    
    ax3.plot(strikes,Df['beta_0.2'],linewidth=3.0)
    
    ax3.plot(strikes,Df['beta_0.6'],linewidth=3.0)  
    
    ax3.plot(strikes,Df['beta_0.8'],linewidth=3.0)  
    
    ax3.plot(strikes,Df['beta_1'],linewidth=3.0)  
    
    
    
     
    ax3.plot(strikes,Df['Normal implied Vol'],linewidth=3.0)
    
    ax3.plot(strikes,Df['SABR IV'],linewidth=3.0)   
    ax3.plot(strikes,Df['Market_IV'],linewidth=3.0)
    ax3.set_xlabel('strikes')
    ax3.set_ylabel('Implied volatilty')  
    
    plt.legend()
    plt.show()    
           
    print('Below Data frame has all Implied Volatility for different beta')
    print(Df)
    
    print('alpha:',alpha)
    print('rho:',rho)
    print('nu:',nu)




strikes = [1,2,3,4,5,6,7,8]

prices_2year = []

for K in strikes:
    
    T = 2
    
    T_N =10
    
    start = (1 + 2*(T-1)) + 1
    
    end  =  start + 2* T_N
    
    P = 0.5*(sum(LD[ start : end ]))
    
        
    beta = 0.8 
    
    alpha = alpha_values[9] + 0.75*(alpha_values[4]-alpha_values[9])
    
    rho = rho_values[9] + 0.75*(rho_values[4]-rho_values[9])
    
    nu = nu_values[9] + 0.75*(nu_values[4]-nu_values[9])
    
    sigma = SABR(S, K, T, alpha, beta, rho, nu)
    
    
    prices_2year.append(P* Displaced_diffusion(S,K,sigma,T,beta))
    
    
strikes = [1,2,3,4,5,6,7,8]

prices_8year = []

for K in strikes:
    
    T = 8
    
    T_N = 10
    
    start = (1 + 2*(T-1)) + 1
    
    end  =  start + 2* T_N
    
    P = 0.5*(sum(LD[ start : end ]))
    
    
    
    beta =  0.8 
    
    
    alpha = alpha_values[14] + 0.4*(alpha_values[9]-alpha_values[14])
    
    rho = rho_values[14] + 0.4*(rho_values[9]-rho_values[14])
    
    nu = nu_values[9] + 0.4*(nu_values[4]-nu_values[9])
    
    sigma = SABR(S, K, T, alpha, beta, rho, nu)
    
    
    prices_8year.append(P* Displaced_diffusion_put(S,K,sigma,T,beta))    


print("Prices for swaption with 2 year maturity")
print(prices_2year)    

print("Prices for swaption with 8 year maturity")
print(prices_8year)   



 