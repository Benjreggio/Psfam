#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:12:26 2021
@author: noumanbutt
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.special import spherical_jn
from scipy import integrate
from itertools import product,permutations

from scipy import arange, pi, sqrt, zeros
from scipy.special import jv, jvp
from scipy.optimize import brentq
from sys import argv
import time
#import matplotlib.pyplot as plt
from scipy.sparse import  isspmatrix

from angular_integrals import angular_integral_values_2p



import sympy
from sympy import S
from sympy.physics.wigner import wigner_3j
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.wigner import gaunt

# print out sample integrals

### recursive method: computes zeros ranges of Jn(r,n) from zeros of Jn(r,n-1)
### (also for zeros of (rJn(r,n))')
### pros : you are certain to find the right zeros values;
### cons : all zeros of the n-1 previous Jn have to be computed;
### note : Jn(r,0) = sin(r)/r




def Jn(r,n):
  return (sqrt(pi/(2.*r))*jv(n+0.5,r))
def Jn_zeros(n,nt):
  zerosj = zeros((n+1, nt), dtype=float)
  zerosj[0] = arange(1,nt+1)*pi
  points = arange(1,nt+n+1)*pi
  racines = zeros(nt+n, dtype=float)
  for i in range(1,n+1):
    for j in range(nt+n-i):
      foo = brentq(Jn, points[j], points[j+1], (i,))
      racines[j] = foo
    points = racines
    zerosj[i][:nt] = racines[:nt]
  return (zerosj)

def rJnp(r,n):
  return (0.5*sqrt(pi/(2*r))*jv(n+0.5,r) + sqrt(pi*r/2)*jvp(n+0.5,r))
def rJnp_zeros(n,nt):
  zerosj = zeros((n+1, nt), dtype=float)
  zerosj[0] = (2.*arange(1,nt+1)-1)*pi/2
  points = (2.*arange(1,nt+n+1)-1)*pi/2
  racines = zeros(nt+n, dtype=float)
  for i in range(1,n+1):
    for j in range(nt+n-i):
      foo = brentq(rJnp, points[j], points[j+1], (i,))
      racines[j] = foo
    points = racines
    zerosj[i][:nt] = racines[:nt]
  return (zerosj)


def integrand(knp,kn,lp,l,i,j):
    
     
     f = lambda x:(x**(2 + 2*i + 2*j))*Jn(kn*x,l)*Jn(knp*x,lp)
     return f
 
def normed_integrand(knp,kn,lp,l,i,j,norm_lp,norm_l):
    
     
     f = lambda x:(x**(2 + 2*i + 2*j))*(Jn(kn*x,l)/norm_l)*(Jn(knp*x,lp)/norm_lp)
     return f    
    
def radial_integral(f):
    value, err = integrate.quadrature(f,0.0,np.pi)
    return value,err









'''
n = 1 #int(argv[1])  # n'th spherical bessel function
nt = 10 #int(argv[2]) # number of zeros to be computed

dr = 0.01
eps = dr/1000

jnz = Jn_zeros(n,nt)[n]
print(jnz/pi)
r1 = arange(eps,jnz[len(jnz)-1],dr)
jnzp = rJnp_zeros(n,nt)[n]
print(jnzp/pi)
r2 = arange(eps,jnzp[len(jnzp)-1],dr)


plt.plot(r1,Jn(r1,n),'b', r2,rJnp(r2,n),'r')
plt.title((str(nt)+' first zeros'))
plt.legend((r'$j_{'+str(n)+'}(r)$', r'$(rj_{'+str(n)+'}(r))\'$'))
plt.plot(jnz,zeros(len(jnz)),'bo', jnzp,zeros(len(jnzp)),'rd')
plt.axhline(y=0.0, color='black', linestyle='-')
#plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
#plt.xaxis.set_minor_formatter(FormatStrFormatter('%d'))
plt.show()

'''


#k0_val=np.zeros([lmax+1,nmax])
#k1_val=np.zeros([lmax+1,nmax])


#for l in range(0,lmax+1):
    
#    k0_val[l] = Jn_zeros(l,nmax)[l]
#    k1_val[l] = rJnp_zeros(l,nmax)[l]
    
#print(k1_val/np.pi)    




    


    


def radial_integral_values_1p(lmax,nmax,e,norms):
    time0 = time.time()
    vals_i1_j0 = np.zeros((nmax,nmax,lmax+1))  # r_1^2 r_2^2
    vals_i0_j1 = np.zeros((nmax,nmax,lmax-1))  
    vals_i2_j0 = np.zeros((nmax,nmax,lmax+1))  # r_1^4
    vals_i3_j0 =  np.zeros((nmax,nmax,lmax+1)) # r_1^6
    
    
    for m,mp in product(range(0,nmax),repeat=2):
        #if(abs(m-mp) <= 1):
          for l in range(0,lmax+1):
            
            vals_i1_j0[m,mp,l], err = radial_integral(normed_integrand(e[l,mp],e[l,m],l,l,1,0,norms[l,mp],norms[l,m]))
        #else:
        #  continue  
    for m,mp in product(range(0,nmax),repeat=2):
        #if(abs(m-mp) <= 1):
          for l in range(0,lmax+1):
            
            vals_i2_j0[m,mp,l], err = radial_integral(normed_integrand(e[l,mp],e[l,m],l,l,2,0,norms[l,mp],norms[l,m]))
        #else:
        #  continue
    
    
    
    for m,mp in product(range(0,nmax),repeat=2):
        #if(abs(m-mp) <= 1):
          for l in range(0,lmax+1):
            
            vals_i3_j0[m,mp,l], err = radial_integral(normed_integrand(e[l,mp],e[l,m],l,l,3,0,norms[l,mp],norms[l,m]))
        #else:
        #  continue
    
    
    
    
    
    
    
    
    for m,mp in product(range(0,nmax),repeat=2):
        
      #if(abs(m-mp) <= 1):  
        for l in range(0,lmax-1):
        
            vals_i0_j1[m,mp,l], err = radial_integral(normed_integrand(e[l+2,mp],e[l,m],l+2,l,0,1,norms[l+2,mp],norms[l,m]))
            
      #else:
       # continue
            
    time1 = time.time()
    print("Time for computing radial integrals: ",time1-time0)                
    return vals_i1_j0,vals_i0_j1,vals_i2_j0,vals_i3_j0   
        
 

#kinetic term v1

def kinetic_term(l1,l2,l3,n1,n2,n3,e,g):
    
   if( g >= 1.2 ):  
    return 0.5 *(e[l1,n1]**2 + e[l2,n2]**2 + e[l3,n3]**2)
   else:  
    return (e[l1,n1] + e[l2,n2] + e[l3,n3])



#radial term v1


def radial_terms1p(l1,l2,l3,n1,n2,n3,n1p,n2p,n3p,vals_i1_j0,vals_i2_j0,vals_i3_j0,g):
    
    alpha_1 = 2.1810429e-2 ## type B
    val = 0.0
    if(n2 == n2p  and n3 == n3p):
        
      if(g < 1.2):  
       val += -(1./(1./(g**2) + alpha_1)) * 0.5 * (1.5**2) * vals_i1_j0[n1p,n1,l1]
        
      val += - 0.30104661  * vals_i1_j0[n1p,n1,l1]
      
      val += -1.4488847e-3 * vals_i2_j0[n1p,n1,l1]
      
      val +=  4.9676959e-5  * vals_i3_j0[n1p,n1,l1]
      
    return val 
  
     

def radial_terms2p(l1,l2,l3,n1,n2,n3,n1p,n2p,n3p,vals_i1_j0,vals_i2_j0):
   
    val = 0.0
    if(n3 == n3p):
        
      val +=  1.2790086e-2  * vals_i1_j0[n1p,n1,l1]*vals_i1_j0[n2p,n2,l2]
    
      val +=  -5.5172502e-5  * (vals_i2_j0[n1p,n1,l1]*vals_i1_j0[n2p,n2,l2] +   vals_i2_j0[n2p,n2,l2]*vals_i1_j0[n1p,n1,l1])
    
    return val
     
    
def radial_term(l1,l2,l3,n1,n2,n3,n1p,n2p,n3p,vals_i1_j0):
    
    if(n3 == n3p):
        
      return vals_i1_j0[n1p,n1,l1]*vals_i1_j0[n2p,n2,l2]
    
    else:
      return 0.0

#angular term v1

 
            
def angular_term_zsq(l1,n1,l2,n2,l3,n3,l1p,n1p,l2p,n2p,l3p,n3p,vals_i1_j0,vals_i0_j1,vals_zsq):
    
    #z^2 term
      val = 0.0
    
      
      if((l3p == l3) and (n3p == n3)): #sanity check
          
       if((abs(l1p-l1) == 2 or l1p == l1) and (abs(l2p-l2) == 2 or l2p == l2)):
           
           val = vals_zsq[(l1p,l1,l2p,l2,l3)]
           
           if( l1p == l1):
               val *= vals_i1_j0[(n1p,n1,l1)]
           elif(l1p - l1 == 2):
               val *= vals_i0_j1[(n1,n1p,l1)]
           elif(l1p -l1 == -2):  
               val *= vals_i0_j1[(n1p,n1,l1p)]
                   
               
               
               
           if( l2p == l2):
               val *= vals_i1_j0[(n2p,n2,l2)]
           elif(l2p - l2 == 2):
               val *= vals_i0_j1[(n2,n2p,l2)]   
           elif(l2p -l2 == -2):  
               val *= vals_i0_j1[(n2p,n2,l2p)]
                   
       else:
           val *= 0
       '''  
       if((l1p-l1 == 2) and (l2p-l2 == 2)):
          
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_zsq[(l1p,l1,l2p,l2,l3)]   
      
       elif((l1p-l1 == -2) and (l2p-l2 == -2)):
          
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_zsq[(l1p,l1,l2p,l2,l3)]   
                            
       elif((l1p-l1 == 2) and (l2p-l2 == -2)):
          
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_zsq[(l1p,l1,l2p,l2,l3)]   
      
       elif((l1p-l1 == -2) and (l2p-l2 == 2)):
          
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_zsq[(l1p,l1,l2p,l2,l3)]   
           
        
       elif((l1p-l1 == 2) and (l2p == l2)):
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i1_j0[l2p,l2,n2p,n2]*vals_zsq[(l1p,l1,l2p,l2,l3)]                           
                             
       elif((l1p-l1 == -2) and (l2p == l2)):
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i1_j0[l2p,l2,n2p,n2]*vals_zsq[(l1p,l1,l2p,l2,l3)]                           
      
       elif((l2p-l2 == 2) and (l1p == l1)):
        val += vals_i1_j0[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_zsq[(l1p,l1,l2p,l2,l3)]                           
                             
       elif((l2p-l2 == -2) and (l1p == l1)):
        val += vals_i1_j0[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_zsq[(l1p,l1,l2p,l2,l3)]                           
      
       elif((l2p == l2) and (l1p == l1)):
        val += vals_i1_j0[l1p,l1,n1p,n1]*vals_i1_j0[l2p,l2,n2p,n2]*vals_zsq[(l1p,l1,l2p,l2,l3)] 

       else:
        val = 0.0                          
      
                              
       '''
    
      else: 
        val = 0.0
        print("l3' != l3 and n3' != n3")                     
                              
                                 
     
                                  
                                   
                               
                              
                                
                                
      return val

def angular_term_xysq(l1,n1,l2,n2,l3,n3,l1p,n1p,l2p,n2p,l3p,n3p,vals_i1_j0,vals_i0_j1,vals_xysq):
    
    #z^2 term
      val = 0.0
    
      
      if((l3p == l3) and (n3p == n3)): #sanity check
          
          
          
       if((abs(l1p-l1) == 2 or l1p == l1) and (abs(l2p-l2) == 2 or l2p == l2)):
           
           val = vals_xysq[(l1p,l1,l2p,l2,l3)]
           
           if( l1p == l1):
               val *= vals_i1_j0[(n1p,n1,l1)]
           elif(l1p - l1 == 2):
               val *= vals_i0_j1[(n1,n1p,l1)]
           elif(l1p -l1 == -2):  
               val *= vals_i0_j1[(n1p,n1,l1p)]
               
           if( l2p == l2):
               val *= vals_i1_j0[(n2p,n2,l2)]
           elif(l2p - l2 == 2):
               val *= vals_i0_j1[(n2,n2p,l2)] 
           elif(l2p - l2 == -2):
               val *= vals_i0_j1[(n2p,n2,l2p)] 
               
       else:
           val *= 0
          
          
          
      
       ''' 
       if((l1p-l1 == 2) and (l2p-l2 == 2)):
          
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_xysq[(l1p,l1,l2p,l2,l3)]   
      
       elif((l1p-l1 == -2) and (l2p-l2 == -2)):
          
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_xysq[(l1p,l1,l2p,l2,l3)]   
                            
       elif((l1p-l1 == 2) and (l2p-l2 == -2)):
          
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_xysq[(l1p,l1,l2p,l2,l3)]   
      
       elif((l1p-l1 == -2) and (l2p-l2 == 2)):
          
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_xysq[(l1p,l1,l2p,l2,l3)]   
           
        
       elif((l1p-l1 == 2) and (l2p == l2)):
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i1_j0[l2p,l2,n2p,n2]*vals_xysq[(l1p,l1,l2p,l2,l3)]                           
                             
       elif((l1p-l1 == -2) and (l2p == l2)):
        val += vals_i0_j1[l1p,l1,n1p,n1]*vals_i1_j0[l2p,l2,n2p,n2]*vals_xysq[(l1p,l1,l2p,l2,l3)]                           
      
       elif((l2p-l2 == 2) and (l1p == l1)):
        val += vals_i1_j0[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_xysq[(l1p,l1,l2p,l2,l3)]                           
                             
       elif((l2p-l2 == -2) and (l1p == l1)):
        val += vals_i1_j0[l1p,l1,n1p,n1]*vals_i0_j1[l2p,l2,n2p,n2]*vals_xysq[(l1p,l1,l2p,l2,l3)]                           
      
       elif((l2p == l2) and (l1p == l1)):
        val += vals_i1_j0[l1p,l1,n1p,n1]*vals_i1_j0[l2p,l2,n2p,n2]*vals_xysq[(l1p,l1,l2p,l2,l3)] 

       else:
        val = 0.0                          
       '''
                              
     
    
      else: 
        val = 0.0
        print("l3' != l3 and n3' != n3")                     
                              
                                 
     
                                  
                                   
                               
                              
                                
                                
      return val




                         



def get_gauge_invariant_ham(H,M,g,indices,e,vals_i1_j0,vals_i0_j1,vals_i2_j0,vals_i3_j0,vals_zsq,vals_xysq):

   time0 =time.time() 
   alpha_1 = 2.1810429e-2
   alpha_2 = 7.5714590e-3

   
   for i in range(0,M):
       
       
      
       multi_indexi= indices[i]
      
       lps = [multi_indexi[0],multi_indexi[1],multi_indexi[2]]
       nps = [multi_indexi[3],multi_indexi[4],multi_indexi[5]]
       
       l1p = multi_indexi[0]
       l2p = multi_indexi[1]
       l3p = multi_indexi[2]
       n1p = multi_indexi[3]
       n2p = multi_indexi[4]
       n3p = multi_indexi[5] 
       
      
       if(tuple([l1p,n1p]) == tuple([l2p,n2p]) == tuple([l3p,n3p])):

         norm_i = 1./6.
        
       elif((tuple([l1p,n1p]) == tuple([l2p,n2p]) ) or (tuple([l1p,n1p]) == tuple([l3p,n3p])) or (tuple([l2p,n2p]) == tuple([l3p,n3p]))):  
         norm_i = 1./(2.*np.sqrt(3.))
      
       else: 
         norm_i = 1./np.sqrt(6.)
      
      
       for j in range(0,M):
      
          val= 0.0 
          multi_indexj = indices[j]
         
          ls = [multi_indexj[0],multi_indexj[1],multi_indexj[2]]
          ns = [multi_indexj[3],multi_indexj[4],multi_indexj[5]]
         
          l1 = multi_indexj[0]
          l2 = multi_indexj[1]
          l3 = multi_indexj[2]
          n1 = multi_indexj[3]
          n2 = multi_indexj[4]
          n3 = multi_indexj[5]
       
          if(tuple([l1,n1]) == tuple([l2,n2]) == tuple([l3,n3])):

              norm_j = 1./6.
        
          elif((tuple([l1,n1]) == tuple([l2,n2]) ) or (tuple([l1,n1]) == tuple([l3,n3])) or (tuple([l2,n2]) == tuple([l3,n3]))):  
              norm_j = 1./(2.*np.sqrt(3.))
      
          else: 
              norm_j = 1./np.sqrt(6.)   
         
         
          for [a,b,c] in list(permutations([0,1,2])):
              
                  l1p = lps[a]
                  l2p = lps[b]
                  l3p = lps[c]
                  n1p = nps[a]
                  n2p = nps[b]
                  n3p = nps[c]
                  
                  for [k,l,m] in list(permutations([0,1,2])):
              
                     l1 = ls[k]
                     l2 = ls[l]
                     l3 = ls[m]
                     n1 = ns[k]
                     n2 = ns[l]
                     n3 = ns[m]
                  
                  
              
         
         
        
         
                     #if( multi_indexi == multi_indexj):
                     if( (tuple([l1,n1]) == tuple([l1p,n1p])) and (tuple([l2,n2]) == tuple([l2p,n2p])) and (tuple([l3,n3]) == tuple([l3p,n3p]))):
                       val +=   (1./(1./(g**2) + alpha_1)) * kinetic_term(l1,l2,l3,n1,n2,n3,e,g)
                   
                 
                     if((l1 == l1p) and (l2 == l2p) and (l3 ==l3p)):  
                         
                       val += 0.5 * ((1./(g**2)) + alpha_2) * radial_term(l1,l2,l3,n1,n2,n3,n1p,n2p,n3p,vals_i1_j0)
                   
                       val += 0.5 * ((1./(g**2)) + alpha_2) * radial_term(l1,l3,l2,n1,n3,n2,n1p,n3p,n2p,vals_i1_j0)
                       val += 0.5 * ((1./(g**2)) + alpha_2) * radial_term(l2,l3,l1,n2,n3,n1,n2p,n3p,n1p,vals_i1_j0)
                       
                       
                       val += radial_terms2p(l1,l2,l3,n1,n2,n3,n1p,n2p,n3p,vals_i1_j0,vals_i2_j0)
                       val += radial_terms2p(l1,l3,l2,n1,n3,n2,n1p,n3p,n2p,vals_i1_j0,vals_i2_j0)
                       val += radial_terms2p(l2,l3,l1,n2,n3,n1,n2p,n3p,n1p,vals_i1_j0,vals_i2_j0)
                       
                       
                      
                       val += radial_terms1p(l1,l2,l3,n1,n2,n3,n1p,n2p,n3p,vals_i1_j0,vals_i2_j0,vals_i3_j0,g)
                       val += radial_terms1p(l2,l1,l3,n2,n1,n3,n2p,n1p,n3p,vals_i1_j0,vals_i2_j0,vals_i3_j0,g)
                       val += radial_terms1p(l3,l1,l2,n3,n1,n2,n3p,n1p,n2p,vals_i1_j0,vals_i2_j0,vals_i3_j0,g)
                       
                       
                       val += -1.2423581e-3  * vals_i1_j0[n1p,n1,l1]*vals_i1_j0[n2p,n2,l2]*vals_i1_j0[n3p,n3,l3]
                       

                 
                     if((l3 == l3p) and (n3p == n3)):
                       val -= 0.5 * (1./(g**2) + alpha_2) * 3.*angular_term_zsq(l1,n1,l2,n2,l3,n3,l1p,n1p,l2p,n2p,l3p,n3p,vals_i1_j0,vals_i0_j1,vals_zsq)  
                       val -= 0.5 * (1./(g**2) + alpha_2) * 6.*angular_term_xysq(l1,n1,l2,n2,l3,n3,l1p,n1p,l2p,n2p,l3p,n3p,vals_i1_j0,vals_i0_j1,vals_xysq)
         
                     if((l2 == l2p) and (n2p == n2)):
                       val -= 0.5 * (1./(g**2) + alpha_2) * 3.*angular_term_zsq(l1,n1,l3,n3,l2,n2,l1p,n1p,l3p,n3p,l2p,n2p,vals_i1_j0,vals_i0_j1,vals_zsq)  
                       val -= 0.5 * (1./(g**2) + alpha_2) * 6.*angular_term_xysq(l1,n1,l3,n3,l2,n2,l1p,n1p,l3p,n3p,l2p,n2p,vals_i1_j0,vals_i0_j1,vals_xysq)
        
                     if((l1 == l1p) and (n1p == n1)):
                       val -= 0.5 * (1./(g**2) + alpha_2) * 3.*angular_term_zsq(l2,n2,l3,n3,l1,n1,l2p,n2p,l3p,n3p,l1p,n1p,vals_i1_j0,vals_i0_j1,vals_zsq)  
                       val -= 0.5 * (1./(g**2) + alpha_2) * 6.*angular_term_xysq(l2,n2,l3,n3,l1,n1,l2p,n2p,l3p,n3p,l1p,n1p,vals_i1_j0,vals_i0_j1,vals_xysq)
                  
                 
                 
                     else:
                       continue  
                  
                
          H[i,j] += val * (norm_i * norm_j)
               
   time1 = time.time()
   #print("Time for constructing Hamiltonian: ",time1-time0)      
   return H   




def A1p_indices(lmax,nmax):
    Ls=list()
    Ns=list()
    l_vals=[]
    n_vals=[]
    
    time0 = time.time()
    for l1,l2,l3 in product(range(0,lmax+1),repeat= 3):
    
            
            if (abs(l1-l2) <= l3 <= (l1+l2) and (l1%2 == 0) and (l2%2 == 0) and (l3%2== 0)):
               #print("L1 L2 L3 =    ",l1,l2,l3 )
               l_vals=[l1,l2,l3]
               
               Ls.append(l_vals)

    
    
    
    
    for n1,n2,n3 in product(range(0,nmax),repeat= 3):
        n_vals = [n1,n2,n3]
        
        Ns.append(n_vals)
     
     
    
    indices_list=list()
    index=[]
    for ls in Ls:
      
     for ns in Ns:
         
        
         index = [ls[0],ls[1],ls[2],ns[0],ns[1],ns[2]]   
         
        
         
         
         
         
         
         for [a,b,c] in list(permutations([0,1,2])):
           
           
            perm_index = [ls[a],ls[b],ls[c],ns[a],ns[b],ns[c]]
            
            if(perm_index in indices_list):
               break
       
         else:
            indices_list.append(index)
        
        
            
       
    
    time1 = time.time()
    print("Time for computing unpermuted A1+ indices ",time1-time0)
    return indices_list











# build a dictionary of zeros
def zeros_dict(lmax,nmax,flag):
    
    z = np.zeros((lmax+1,nmax))
    if(flag == 1):
      for l in range(0,lmax+1):
        z[l] = Jn_zeros(l,nmax)[l]
    
    elif(flag == 0):
        for l in range(0,lmax+1):
          z[l] = rJnp_zeros(l,nmax)[l]
    return z/np.pi

def radial_func_norms(lmax,nmax,e):
    
    norms = np.zeros((lmax+1,nmax))
    
    for l in range(0,lmax+1):
        for n in range(0,nmax):
            
            val ,err = radial_integral(integrand(e[l,n],e[l,n],l,l,0,0))
            norms[l,n] = np.sqrt(val)
    return norms

def energies(indices_list,e,g):
    energies =list()
     
    for i in range(0,len(indices_list)):
        multi_indexi = indices_list[i]
       
        l1=multi_indexi[0]
        l2=multi_indexi[1]
        l3=multi_indexi[2]
        n1=multi_indexi[3]
        n2=multi_indexi[4]
        n3=multi_indexi[5]
        
        if( g < 1.2):
            
         E_i = (e[l1,n1] + e[l2,n2] + e[l3,n3])                 ## type B
        else:
         E_i = 0.5 * (e[l1,n1]**2 + e[l2,n2]**2 + e[l3,n3]**2) ## type A
        #print("Index Energy    ", multi_indexi,E_i)
        energies.append(E_i)
         
        
        
    return energies  

def sorted_indices(Es,indices_list,M):
    sorted_indices=list()
    sort_index = np.argsort(Es)
    #sort_energies = np.sort(Es)
    #print(sort_energies[:M])
   
    sorted_indices = list()
    for index in sort_index:
        
       
       sorted_indices.append(indices_list[index])
    
     
    
    
    return sorted_indices[:M]


def simplify(H,M):
    c = 0.2
    for i in range(M):
        for j in range(M):
            if( i != j):
             if( abs(H[i][j])/abs(H[i][i]- H[j][j]) <  c):
                H[i][j] = 0.0
             else:
                 continue
            else:
                continue
    return H            
                

          
 
def get_eigenvalues(g,M,lmax,nmax,flag):
    
    #e=zeros_dict(lmax,nmax,flag)
    
    #with open('epsilon_typeB_e1_omega_1.5, 'rb') as fp:
         #pickle.dump(e,fp)
         #e  = pickle.load(fp).reshape(21,10)
    if( g >= 1.2):
      with open('epsilon_e0', 'rb') as fp:
         
         e  = pickle.load(fp)
    
    else:
      with open('epsilon_e0_typeB', 'rb') as fp:
         
         e  = pickle.load(fp).reshape((21,10)) 
        
        
      #e  = np.loadtxt("epsilon_typeB_e_1_omega_1.5.csv", delimiter=',').reshape((21,10))    
    
    
    
    
    
    with open('A1+_indices', 'rb') as fp:
     indices = pickle.load(fp)
   
    
    Es = energies(indices, e,g)
    
    indices_sorted = sorted_indices(Es, indices, M)
    
    
    
    
    print("------------------------------------")
   
    if( g >= 1.2):
       
     with open('F_i1_j0_e0', 'rb') as fp:
        vals_i1_j0 = pickle.load(fp)
     with open('F_i0_j1_e0', 'rb') as fp:
        vals_i0_j1 = pickle.load(fp)
     with open('F_i2_j0_e0', 'rb') as fp:
        vals_i2_j0 = pickle.load(fp)    
     with open('F_i3_j0_e0', 'rb') as fp:
        vals_i3_j0 = pickle.load(fp)        
     '''
     with open('F_i1_j0_e1', 'rb') as fp:
        vals_i1_j0 = pickle.load(fp)
     with open('F_i0_j1_e1', 'rb') as fp:
        vals_i0_j1 = pickle.load(fp)
     with open('F_i2_j0_e1', 'rb') as fp:
        vals_i2_j0 = pickle.load(fp)    
     with open('F_i3_j0_e1', 'rb') as fp:
        vals_i3_j0 = pickle.load(fp)
     '''
   
    
    
            
    
   
    if(g < 1.2):
        
      with open('F_i1_j0_e0_typeB', 'rb') as fp:
         vals_i1_j0 = pickle.load(fp)
        
      with open('F_i0_j1_e0_typeB', 'rb') as fp:
         vals_i0_j1 = pickle.load(fp)
        
      with open('F_i2_j0_e0_typeB', 'rb') as fp:
         vals_i2_j0 = pickle.load(fp)  
        
    
      vals_i3_j0  = np.loadtxt("flatFi3j0List.csv", delimiter=',').reshape((10, 10, 21))
      '''
      vals_i0_j1  = np.loadtxt("flat_F_i0_j1_e1_typeB_omega1.5.csv", delimiter=',').reshape((10, 10, 19))
      vals_i1_j0  = np.loadtxt("flat_F_i1_j0_e1_typeB_omega1.5.csv", delimiter=',').reshape((10, 10, 21))
      vals_i2_j0  = np.loadtxt("flat_F_i2_j0_e1_typeB_omega1.5.csv", delimiter=',').reshape((10, 10, 21))
      vals_i3_j0  = np.loadtxt("flat_F_i3_j0_e1_typeB_omega1.5.csv", delimiter=',').reshape((10, 10, 21))
      '''
    
    
    f1=open("Vals_Zsq2p","rb")
    
    f2=open("Vals_XYsq2p","rb")
    
    vals_zsq = pickle.load(f1)
    
    vals_xysq = pickle.load(f2)
    
   
    
    H = np.zeros((M,M))
    
    H = get_gauge_invariant_ham(H, M, g, indices_sorted, e, vals_i1_j0, vals_i0_j1,vals_i2_j0,vals_i3_j0,vals_zsq, vals_xysq)
    
    #H = simplify(H,M)
    
    evals, evecs = np.linalg.eigh(H)
    
    #print(isspmatrix(H))
    
    return H, evals
       





