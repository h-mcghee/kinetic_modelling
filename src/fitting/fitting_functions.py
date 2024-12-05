import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import optimize
from scipy.signal import convolve
from lmfit import Model
from lmfit.models import StepModel
from lmfit import Parameters
import os
import matplotlib.ticker as ticker


class FittingFunctions:
    def __init__(self):
        self.step_model = StepModel(form = 'erf')

    def conv(self,x,t0,sigma, *args):
        num_exponentials = len(args) // 2
        num_params = len(args)

        if num_params % 2 != 0 or num_exponentials == 0:
            raise ValueError("The number of parameters must be a multiple of 2 (A, tau) for each exponential.")

        # sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        result = np.zeros_like(x)

        for i in range(num_exponentials):
            A = args[2*i]
            tau = args[2*i + 1]
            exp_term = (A / 2) * np.exp((-1 / tau) * (x - t0)) * np.exp((sigma**2) / (2 * tau**2)) * \
            (1 + scipy.special.erf((x - t0 - ((sigma **2) / tau)) / (np.sqrt(2.0) * sigma)))
            result += exp_term

        return result

    
    exp1 = lambda self, x, t0, sigma, A1, tau1: self.conv(x, t0, sigma, A1, tau1)
    exp2 = lambda self, x, t0, sigma, A1, tau1,A2,tau2: self.conv(x, t0, sigma, A1, tau1,A2,tau2)
    exp3 = lambda self, x, t0, sigma, A1, tau1,A2,tau2,A3,tau3: self.conv(x, t0, sigma, A1, tau1,A2,tau2,A3,tau3)

    def gaussian(self,x, A, t0, sigma):
        # sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        return A*np.exp(-((x - t0) ** 2) / (2 * sigma ** 2))


    def delay_decay(self,x,A,tau,tdelay):
        return A*np.exp(-(x-tdelay)/(tau))

    def exp_growth(self,x,A,tdelay,tau):
        def heaviside(x, A, x0):
            return A * (x > x0)
        
        H = heaviside(x,1,tdelay)
        return A*(1-np.exp(-(x-tdelay)/(tau))) * H

    def step(self, x, amplitude, center, sigma):
        return self.step_model.func(x, amplitude, center, sigma,form = 'erf')

    def heaviside(self, x, A, x0):
        return A * (x > x0)

    def exp_decay(self,x,A,tau,tdelay):
        H = self.heaviside(x,1,tdelay)
        return A*np.exp(-(x-tdelay)/(tau)) * H

    def irf(self,x, t0, sigma):
        # sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-((x - t0) ** 2) / (2 * sigma ** 2))

    def step_conv(self,x,t0,s,amplitude,center,sigma):
        dt = x[1] - x[0]
        step_signal = self.step(x,amplitude,center,sigma)
        irf_signal = self.irf(x,t0,s)
        conv_result = convolve(step_signal, irf_signal, mode='full') * dt
        x_conv = np.linspace(x[0],x[0]+dt*(len(conv_result)-1),len(conv_result))
        conv_result_interp = np.interp(x, x_conv, conv_result)
        conv_result /= np.trapz(conv_result_interp, x)
        return conv_result_interp
    
    def composite(self,x,t0,sigma,A,tau,tdelay,A_g):
        return self.num_conv(x,t0,sigma,A,tau,tdelay)+self.gaussian(x,A_g,t0,sigma)

    def compsite2(self,x,t0,sigma,A,tau,A_e,tdelay,tau_e,A_g):
        return self.exp1(x,t0,sigma,A,tau)+self.exp_growth(x,A_e, tdelay, tau_e)+self.gaussian(x,A_g,t0,sigma)

    def sequential(self,x,A,A1,tau1,A2,tau2,t0,sigma):
        H = self.heaviside(x,1,t0)
        comp1_decay = self.exp_decay(x,A1,tau1)
        comp2_decay = self.exp_decay(x,A2,tau2)
        comp2_growth = self.exp_growth(x,A2,tau2,0)
        decay_signal = (comp1_decay+comp2_decay*comp2_growth)*H
        return A*decay_signal

    def seq(self,x,A,tau1,tau2,tdelay,sigma):
        h = np.linspace(x[0],x[-1],10000)
        irf = self.irf(h,0,sigma)
        H = self.heaviside(x,1,tdelay)
        k1 = 1/tau1
        k2 = 1/tau2
        comp = A * (k1/(k2-k1))*(np.exp(-k1*x) - np.exp(-k2*x)) * H
        conv = np.convolve(comp,irf,mode = 'full') /np.sum(irf)
        newx = np.linspace(h[0] + h[0], h[-1] + h[-1], len(conv))
        conv_interp = np.interp(x, newx, conv)
        return conv_interp

    def roman(self,x,A,tau1,tau2,t0,sigma,tdelay):
        k1 = 1/tau1
        k2 = 1/tau2
        return A * (((np.exp(-(x-t0-tdelay)*k2)*np.exp(0.5*((sigma*k2))**2)*(1+scipy.special.erf(((x-tdelay)-t0-((sigma**2*k2)))/(np.sqrt(2)*sigma)))) - (np.exp(-(x-tdelay-t0)*k1)*np.exp(0.5*((sigma*k1))**2)*(1+scipy.special.erf(((x-tdelay)-t0-((sigma**2)*k1))/(np.sqrt(2)*sigma))))))
        
    def error(self,x, A, t0, sigma):
        return (A/2) * (1 + scipy.special.erf((x - t0) / (np.sqrt(2) * sigma)))

    # def exp_offset(self,x,A,t0,sigma,tau,c):
    #     return self.exp1(x,t0,sigma,A,tau) + (c*self.error(x,A,t0,sigma))

    def roman_test(self,x,A,tau1,tau2,t0,sigma,tdelay):
        k1 = 1/tau1
        k2 = 1/tau2
        return A * (((np.exp(-(x-t0-tdelay)*k2)*np.exp(0.5*((sigma*k2))**2)*(1+scipy.special.erf(((x)-t0-((sigma**2*k2)))/(np.sqrt(2)*sigma)))) - (np.exp(-(x-tdelay-t0)*k1)*np.exp(0.5*((sigma*k1))**2)*(1+scipy.special.erf(((x)-t0-((sigma**2)*k1))/(np.sqrt(2)*sigma))))))
        
    def roman2(self,x,A,tau1,tau2,tau3,t0,sigma,tdelay):
        k1 = 1/tau1
        k2 = 1/tau2
        k3 = 1/tau3

        return A*((np.exp(-(x-t0-tdelay)*k3)*np.exp(0.5*((sigma*k3)**2))*((k1-k2)/((k1-k3)*(k2-k3))) * (1 + scipy.special.erf((x-t0-tdelay-((sigma**2)*k3))/(np.sqrt(2)*sigma)))) - \
        (np.exp(-(x-t0-tdelay)*k2)*((np.exp(0.5*((sigma*k2)**2)))/(k2-k3)) * (1 + scipy.special.erf((x-t0-tdelay-((sigma**2)*k2))/(np.sqrt(2)*sigma)))) + \
        (np.exp(-(x-t0-tdelay)*k1)*((np.exp(0.5*((sigma*k1)**2)))/(k1-k3)) * (1 + scipy.special.erf((x-t0-tdelay-((sigma**2)*k1))/(np.sqrt(2)*sigma))))) 

    def sequential2(self,x,A,t0,sigma,tau1,tau2):
        k1 = 1/tau1
        k2 = 1/tau2
        return (k1/(k2-k1))*(self.exp1(x,t0,sigma,A,tau1) - self.exp1(x,t0,sigma,A,tau2)) #A parameter including in single exp functions

    def sequential3(self,x,A,t0,sigma,tau1,tau2,tau3):
        k1 = 1/tau1
        k2 = 1/tau2
        k3 = 1/tau3

        return (k1*k2)* ( (self.exp1(x,t0,sigma,A,tau1)/((k2-k1)*(k3-k1))) \
            + (self.exp1(x,t0,sigma,A,tau2))/ ((k1-k2)*(k3-k2)) \
                + (self.exp1(x,t0,sigma,A,tau3))/ ((k1-k3)*(k2-k3)) )

    def seq3nog(self,x,A,tau1,tau2):
        k1 = 1/tau1
        k2 = 1/tau2
        return (x>0)*A*(1 - ((1/(k2-k1))*(k2*np.exp(-k1*x) - k1*np.exp(-k2*x)))) 

    def russell(self,x,A,tau,tdelay,sigma):
        # H = self.heaviside(x,1,tdelay)
        #define dummy axis
        h = np.linspace(x[0],x[-1],10000)
        irf = self.irf(h,0,sigma)
        comp = self.exp_growth(h,A,tdelay,tau)
        conv = np.convolve(comp,irf,mode = 'full') /np.sum(irf)
        newx = np.linspace(h[0] + h[0], h[-1] + h[-1], len(conv))
        conv_interp = np.interp(x, newx, conv)
        mask = x>8000
        conv_interp[mask] = np.max(conv_interp)
        return conv_interp


    def num_conv(self,x,sigma,A,tau,tdelay):
        h = np.linspace(x[0],x[-1],10000)
        irf = self.irf(h,0,sigma)
        comp = self.exp_decay(h,A,tau,tdelay)
        conv = np.convolve(comp,irf,mode = 'full') /np.sum(irf)
        newx = np.linspace(h[0] + h[0], h[-1] + h[-1], len(conv))
        conv_interp = np.interp(x, newx, conv)
        return conv_interp




def conv(x,t0,sigma, *args):
    num_exponentials = len(args) // 2
    num_params = len(args)

    if num_params % 2 != 0 or num_exponentials == 0:
        raise ValueError("The number of parameters must be a multiple of 2 (A, tau) for each exponential.")

    # sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    result = np.zeros_like(x)

    for i in range(num_exponentials):
        A = args[2*i]
        tau = args[2*i + 1]
        exp_term = A / 2 * np.exp((-1 / tau) * (x - t0)) * np.exp((sigma**2) / (2 * tau**2)) * \
        (1 + scipy.special.erf((x - t0 - ((sigma **2) / tau)) / (np.sqrt(2.0) * sigma)))
        result += exp_term

    return result

def exp_growth(x,A,tau,t0):
    return A*(1-np.exp(-x/(tau-t0)))
