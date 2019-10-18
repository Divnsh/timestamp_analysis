#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:42:54 2018

@author: divyansh
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

# Reading data
#fdata = pd.read_csv('time_campgn.csv',low_memory=False)
#print(fdata.head(10))

import warnings
warnings.filterwarnings("ignore")

# Simulating data
time_slot = np.zeros(1000)
hit_rates = np.zeros(1000)
mode = np.zeros(1000)
offer = np.zeros(1000)

for i in range(0,1000):
  time_slot[i] = np.random.randint(1,7)
  hit_rates[i] = np.random.randint(0,5000)*.01
  mode[i] = np.random.randint(1,4)
  offer[i] = np.random.randint(1,4)


fdata = pd.DataFrame({'Time_slot':time_slot, 'Hit_rates':hit_rates, 'Mode':mode, 'Offer_type':offer})
fdata = fdata[['Time_slot', 'Mode', 'Offer_type', 'Hit_rates']]
print(fdata.head(10))


# Mean and Std dev for each time slot
m = fdata.groupby('Time_slot').mean()['Hit_rates']
print("Means per time slot\n",m)

std = fdata.groupby('Time_slot').std()['Hit_rates']
print("\nStandard deviations per time slot\n",std)

counts = fdata.groupby('Time_slot').size()
counts.plot(kind = 'bar')
plt.savefig('counts.png',bbox_inches='tight')

## 95% and 99% Confidence intervals for mean hit rates assuming normality
l95 = []
u95 = []
l99 = []
u99 = []
for i in range(1,7):
    l95.append(m[i]-1.98*std[i]/np.sqrt(counts[i]))
    u95.append(m[i]+1.98*std[i]/np.sqrt(counts[i]))
    l99.append(m[i]-2.58*std[i]/np.sqrt(counts[i]))
    u99.append(m[i]+2.58*std[i]/np.sqrt(counts[i]))
    
ci = pd.DataFrame({'Time_slot':[1,2,3,4,5,6],'lower95':l95,'upper95':u95,'lower99':l99,'upper99':u99})    
print("\n95% and 99% Confidence intervals for mean hit rates assuming normality: \n",ci)

## Plotting distribution plots
pl = sns.FacetGrid(fdata, col = "Time_slot")
pl.map(sns.distplot, 'Hit_rates')
plt.savefig('dists.png', bbox_inches='tight')

# Segment by Offer type
plo = sns.FacetGrid(fdata, col = "Time_slot", hue="Offer_type")
plo.map(sns.distplot, 'Hit_rates')
plo.add_legend()
plt.savefig('dists_offers.png', bbox_inches='tight')

# Segment by Mode of communication
plm = sns.FacetGrid(fdata, col = "Time_slot", hue="Mode")
plm.map(sns.distplot, 'Hit_rates')
plm.add_legend() 
plt.savefig('dists_modes.png', bbox_inches='tight')

# Comparing all distributions
for i in range(1,7):
    t = fdata[(fdata['Time_slot']==i)]['Hit_rates']
    print("\n\nTime slot = ",i,"\n",t.describe())
        
plb = sns.FacetGrid(fdata, col = "Time_slot")
plb.map(sns.boxplot, 'Hit_rates', orient='v')
plt.savefig('boxplots.png', bbox_inches='tight')

# ANOVA tests - 
m = smf.ols('Hit_rates~C(Mode)',data=fdata).fit() #Mode of communication
print("Association between Mode of communication and hit rates\n",m.summary())
if(m.f_pvalue <0.05):
        mc = multi.MultiComparison(fdata['Hit_rates'], fdata['Mode'])
        res = mc.tukeyhsd()
        print("\nThere is a significant association between Hit rates and mode of communication. Here is the post-hoc test for mode of communication: \n",res.summary())


o = smf.ols('Hit_rates~C(Offer_type)', data=fdata).fit() #Offer type
print("\nAssociation between offer type and hit rates\n",o.summary())
if(o.f_pvalue <0.05):
        Mc = multi.MultiComparison(fdata['Hit_rates'], fdata['Mode'])
        Res = Mc.tukeyhsd()
        print("\nThere is a significant association between Hit rates and mode of communication. Here is the post-hoc test for mode of communication: \n",Res.summary())

# within each time slot 
for i in range(1,7):
    fd = fdata[fdata['Time_slot']==i]
    mw = smf.ols('Hit_rates~C(Mode)',data=fd).fit() #Mode of communication
    ow = smf.ols('Hit_rates~C(Offer_type)',data=fd).fit() #Offer type
    print("\nTest results for time slot ",i," \nMode of communication:\n",mw.summary(),"\nOffer type:\n", ow.summary())
    if(mw.f_pvalue <0.05):
        mc1 = multi.MultiComparison(fd['Hit_rates'], fd['Mode'])
        res1 = mc1.tukeyhsd()
        print("\nThere is a significant association between Hit rates and mode of communication. Here is the post-hoc test for mode of communication: \n",res1.summary())
        
    if(ow.f_pvalue <0.05):
        mc2 = multi.MultiComparison(fd['Hit_rates'], fd['Offer_type'])
        res2 = mc2.tukeyhsd()
        print("\nThere is a significant association between Hit rates and offer type. Post-hoc test for offer type: \n",res2.summary())
    

## Fitting distributions using Expectation Maximization algorithm for each time slot
from sklearn.mixture import GMM

for j in range(1,7):

        hr = fdata[fdata['Time_slot']==j]['Hit_rates']
        hr = hr.reshape(-1,1)

        N = np.arange(1, 5) #Taking a max of 4 components
        models = [None for i in range(len(N))]
        
        for i in range(len(N)):
            models[i] = GMM(N[i], n_iter=5).fit(hr) #Iterate from 1 to 4 components
        
        # compute the AIC and the BIC
        AIC = [m.aic(hr) for m in models] #AIC scores for all models
        BIC = [m.bic(hr) for m in models] #BIS scores for all models  
        
        # Plotting the results
        #  We'll use three panels:
        #   1) data + best-fit mixture
        #   2) AIC and BIC vs number of components
        #   3) probability that a point came from each component
        
        fig = plt.figure(figsize=(8, 4), dpi=80)
        fig.subplots_adjust(left=0.12, right=0.97,
                            bottom=0.21, top=0.9, wspace=0.5)
        
        
        # plot 1: data + best-fit mixture
        ax = fig.add_subplot(131)
        M_best = models[np.argmin(AIC)]
        
        x = np.linspace(-1, 50, 1000)
        logprob, responsibilities = M_best.score_samples(x.reshape((-1,1)))
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        
        ax.hist(x, 30, normed=True, histtype='stepfilled', alpha=0.4)
        ax.plot(x, pdf, '-k')
        ax.plot(x, pdf_individual, '--k')
        ax.text(0.04, 0.96, "Best-fit Mixture",
                ha='left', va='top', transform=ax.transAxes)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p(x)$')
        
        
        # plot 2: AIC and BIC
        ax = fig.add_subplot(132)
        ax.plot(N, AIC, '-k', label='AIC')
        ax.plot(N, BIC, '--k', label='BIC')
        ax.set_xlabel('n. components')
        ax.set_ylabel('information criterion')
        ax.legend(loc=2)
        
        
        # plot 3: posterior probabilities for each component
        ax = fig.add_subplot(133)
        
        p = M_best.predict_proba(x.reshape((-1,1)))
          
        p = p.cumsum(1).T
        
        ax.fill_between(x, 0, p[0], color='gray', alpha=0.3)
        ax.fill_between(x, p[0], p[1], color='gray', alpha=0.5)
        ax.fill_between(x, p[1], 1, color='gray', alpha=0.7)
        ax.set_xlim(-1, 50)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$p({\rm class}|x)$')
        
        ax.text(0, 0.3, 'class 1', rotation='vertical')
        ax.text(45, 0.3, 'class 2', rotation='vertical')
        
        plt.xlabel("Means of fitted gaussian components: %s" %np.array2string(M_best.means_, separator=','))
        plt.title("Time slot %i" %j, loc = 'left', fontsize=20)
        plt.savefig('time%i.png'%j, bbox_inches='tight')
        plt.show()
        print("\n\nTime slot %i"%j)
        print("\nMeans of fitted gaussian components : ",M_best.means_)
        print("\nVariance of fitted gaussian components : ",M_best.covars_)
        print("\nStandard deviation of fitted gaussian components : ",np.sqrt(M_best.covars_))        
        sample = []                # QQ Plot
        for l in range(0,500):     # 500 generated samples from fitted model
            for m in range(0,len(hr)):
               k =  np.random.choice(M_best.n_components,p = M_best.weights_)
               sample.append(np.random.normal(M_best.means_[k], np.sqrt(M_best.covars_[k])))    
        sample = np.concatenate(sample) 
        sm.qqplot_2samples(sample, np.repeat(np.concatenate(hr),500), ylabel="quantiles (generated hit rates)", xlabel = "quantiles (original hit rates)", line='45')
        plt.title("Time Slot %i: QQ Plot between generated sample from fitted distribution and original hit rates"%j)
        plt.savefig("qqplot%i.png"%j)
        plt.show()
       
        
        
# Empirical confidence intervals using bootstrap sampling
xbar = []
sstd = []
lower95mean=[]
upper95mean=[]
lower99mean=[]
upper99mean=[]
lower95sd=[]
upper95sd=[]
lower99sd=[]
upper99sd=[]
bootstraps = 10000
for i in range(1,7):
    fb = fdata[fdata['Time_slot']==i]['Hit_rates']
    for i in range(1,bootstraps):
        s = np.random.choice(fb, size=fb.shape, replace=True)
        xbar.append(s.mean())
        sstd.append(s.std())
    lower95mean.append(np.percentile(xbar,2.5))   
    lower95sd.append(np.percentile(sstd,2.5))
    upper95mean.append(np.percentile(xbar,97.5))
    upper95sd.append(np.percentile(sstd,97.5))
    lower99mean.append(np.percentile(xbar,0.5))   
    lower99sd.append(np.percentile(sstd,0.5))
    upper99mean.append(np.percentile(xbar,99.5))
    upper99sd.append(np.percentile(sstd,99.5))
    xbar=[]
    sstd=[]

boot_mean_ci = pd.DataFrame({'Time_slot':[1,2,3,4,5,6], 'lower95mean':lower95mean,'upper95mean':upper95mean, 'lower99mean':lower99mean, 'upper99mean':upper99mean})
boot_std_ci = pd.DataFrame({'Time_slot':[1,2,3,4,5,6], 'lower95sd':lower95mean,'upper95sd':upper95mean, 'lower99sd':lower99mean, 'upper99sd':upper99mean})

print("\n95% and 99% bootstrap confidence intervals (mean): \n",boot_mean_ci)
print("\n95% and 99% bootstrap confidence intervals (standard deviation): \n",boot_std_ci)
    
# Outputting in a text file
import subprocess
with open("output_campaign.txt", "w+") as output:
    subprocess.call(["python", "./script.py"], stdout=output)     