# second order MF for cortical RS and FS cell populations

# imports
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc
from mytools import progressBar,double_gaussian,ornstein_uhlenbeck

#====functions==============================

# the transfer function
def TF(typ,fexc,finh,adapt):
    '''
    transfer function for RS and FS cells...
    typ :: RS or FS to load the correct fitting parameter data and bio cell variables
    fexc and finh :: total input excitatory and inhibitory frequencies to the population
    adapt :: the adaptation w for the population
    '''
    # change here the cell parameters for both cells:
    if typ=='RS':
        P = PRS
        Nexc=400
        Ninh=100
        Qe=1.5e-9
        Qi=5e-9
        Cm=200e-12
        El=-63e-3
    elif typ=='FS':
        P = PFS
        Nexc=400
        Ninh=100
        Qe=1.5e-9
        Qi=5e-9
        Cm=200e-12
        El=-67e-3

    # to ensure that the input frequencies are not negative or too small
    if fexc<1e-9: fe=1e-9
    else: fe = fexc*Nexc
    if finh<1e-9: fi=1e-9
    else: fi = finh*Ninh

    #-----start of the analytical formulation of the membrane potential flucktuations------
    # or MPF in short
    muGi = Qi*Ti*fi;
    muGe = Qe*Te*fe;
    muG = Gl+muGe+muGi;
    muV = (muGe*Ee+muGi*Ei+Gl*El - adapt)/muG;
    # muV = (muGe*Ee+muGi*Ei+Gl*El - fout*Tw*b + a*El)/(muG+a);
    
    
    muGn = muG/Gl;
    Tm = Cm/muG;
    
    Ue =  Qe/muG*(Ee-muV);
    Ui = Qi/muG*(Ei-muV);

    sV = np.sqrt(fe*(Ue*Te)*(Ue*Te)/2./(Te+Tm)+fi*(Ui*Ti)*(Ui*Ti)/2./(Ti+Tm));

    Tv = ( fe*(Ue*Te)*(Ue*Te) + fi*(Qi*Ui)*(Qi*Ui)) /( fe*(Ue*Te)*(Ue*Te)/(Te+Tm) + fi*(Qi*Ui)*(Qi*Ui)/(Ti+Tm) );
    TvN = Tv*Gl/Cm;
    #-----end of the MPF--------------------------------------------------------------------
    
    # normalising the MPF variables to ensure a better comparison to other fits
    muV0=-60e-3;
    DmuV0=10e-3;
    sV0=4e-3;
    DsV0=6e-3;
    TvN0=0.5;
    DTvN0=1.;

    # the phenomenological function for Vthre with the parameters fitted on numerical data
    vthre = P[0] + P[1]*(muV-muV0)/DmuV0 + P[2]*(sV-sV0)/DsV0 + P[3]*(TvN-TvN0)/DTvN0 \
        + P[4]*((muV-muV0)/DmuV0)*((muV-muV0)/DmuV0) + P[5]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0 + P[6]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0 \
        + P[7]*((sV-sV0)/DsV0)*((sV-sV0)/DsV0) + P[8]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0 + P[9]*((TvN-TvN0)/DTvN0)*((TvN-TvN0)/DTvN0);

    # the actual transfer function
    frout = 1/(2*Tv) * erfc( (vthre - muV)/(np.sqrt(2)*sV) )
    if frout<1e-9: frout=1e-9 # exclude negative outputs..
    
    return frout;

# the differential equation for the change of adaptation.. for both cell types
def MFw(typ, adapt, fexc, finh):
    if typ=='FS':
        # b=0 right now.. write b instead of the 0 below if you want to add adaptation for the FS cell
        adapt = -adapt/Tw+0*finh
    # change here also the cell parameters if you changed them in the TF function!!
    if typ=='RS':
        a=4e-9
        Nexc=400
        Ninh=100
        Qe=1.5e-9
        Qi=5e-9
        El=-63e-3

        fe=fexc*Nexc
        fi=finh*Ninh
        
        muGi = Qi*Ti*fi;
        muGe = Qe*Te*fe;
        muG = Gl+muGe+muGi;
        muV = (muGe*Ee+muGi*Ei+Gl*El - adapt)/muG;
        adapt = -adapt/Tw+b*fexc + a*(muV-El)/Tw
        
    return adapt



#----constants (which are the same for both populations!!)----------------
Gl=10*1.e-9
Tw=500*1e-3
b=40*1e-12

Ti=5*1.e-3 # Tsyn FS
Te=5*1.e-3 # Tsyn RS
Ee=0
Ei=-80*1.e-3
#-------------------------------------------------------------------------


# load fitting parameters...
PRS=np.load("C:\VSCode\DB_comparison\mf\data\RS-cell_CONFIG1_fit.npy")[[0,1,2,3,5,8,9,6,10,7]]
PFS=np.load("C:\VSCode\DB_comparison\mf\data\FS-cell_CONFIG1_fit.npy")[[0,1,2,3,5,8,9,6,10,7]]


#---------MF and numerics params----------
T=10e-3

tfinal=1 # s
dt=1e-4 # s
df=1e-5 # Hz
tsteps=int(tfinal/dt)

t=np.linspace(0, tfinal, tsteps)


#-----external inputs----------------------------------------------------
# uncomment and comment the ones you want to use/not use..

#=CORTEX (Peripherie?)
#-constant
external_input=np.full(tsteps, 2.5)
# external_input+=np.random.randn(tsteps)/2

#-timeframe
# external_input=np.zeros(tsteps)
# external_input[:200] = 1
# external_input[int(tsteps*2/4):int(tsteps*3/4)] = 2

#-sinus
# ampl=.6
# fFSq=1
# external_input = ampl/2*(1-np.cos(fFSq*2*np.pi*t))

#-noise?
# external_input=ornstein_uhlenbeck(tsteps,tfinal, 10, .5, 2, start=0,seed=20)
# external_input[external_input<0]=0

#=STIM (just going into the excitatory RS cell population!!)
stim=np.zeros(tsteps)
# stim=double_gaussian(t, 1, 0.01, 0.2, 5)
# stim[int(tsteps*2/4):int(tsteps*3/4)] = 20.
# stim[int(tsteps/2):int(tsteps/2)+50] = 20


#----initial conditions---------------------------------------------------
fecont=1;
ficont=10;
# we=fecont*b*Tw
# wi=ficont*b*Tw
# we=MFw('RS', 0, fecont, 0)*Tw
# wi=MFw('FS', 0, external_input[0]+fecont/16, ficont)*Tw
# cee,cei,cii=.5,.5,.5
we,wi=1e-10,0
#-------------------------------------------------------------------------

# create empty lists for the variables we want to keep track of in the integration loop
LSwe,LSwi=[],[]
LSfe,LSfi=[],[]
LScee,LScii=[],[]
test,test2=[],[]

#======the MAIN loop for integration==============================================
for i in progressBar(range(len(t))):
    
    fecontold=fecont
    ficontold=ficont
    weold,wiold=we,wi
    # ceeold,ceiold,ciiold=cee,cei,cii
    # this are the excitatory inputs to both populations:
    RSfe = external_input[i]+fecont+stim[i]
    FSfe = external_input[i]+fecont

    #-TFs
    Fe = TF('RS',RSfe,ficont,we)
    Fi = TF('FS',FSfe,ficont,wi)


    #-----TF derivatives------------------------------------------------------------

    # #-first order
    # dveFe = (TF('RS',RSfe+df/2,ficont,we)-TF('RS',RSfe-df/2,ficont,we))/df
    # dviFe = (TF('RS',RSfe,ficont+df/2,we)-TF('RS',RSfe,ficont-df/2,we))/df
    # dveFi = (TF('FS',FSfe+df/2,ficont,wi)-TF('FS',FSfe-df/2,ficont,wi))/df
    # dviFi = (TF('FS',FSfe,ficont+df/2,wi)-TF('FS',FSfe,ficont-df/2,wi))/df

    # #-second order
    # dvedveFe = ( TF('RS',RSfe+df,ficont,we) - 2*Fe + TF('RS',RSfe-df,ficont,we) )/df**2
    # dvidveFe = ( (TF('RS',RSfe+df/2,ficont+df/2,we)-TF('RS',RSfe-df/2,ficont+df/2,we))\
    #             - (TF('RS',RSfe+df/2,ficont-df/2,we)-TF('RS',RSfe-df/2,ficont-df/2,we)) )/df**2
    # dvidviFe = ( TF('RS',RSfe,ficont+df,we) - 2*Fe + TF('RS',RSfe,ficont-df,we) )/df**2
    # dvedviFe = ( (TF('RS',RSfe+df/2,ficont+df/2,we)-TF('RS',RSfe+df/2,ficont-df/2,we))\
    #             - (TF('RS',RSfe-df/2,ficont+df/2,we)-TF('RS',RSfe-df/2,ficont-df/2,we)) )/df**2
    # dvedveFi = ( TF('FS',FSfe+df,ficont,wi) - 2*Fi + TF('FS',FSfe-df,ficont,wi) )/df**2
    # dvidveFi = ( (TF('FS',FSfe+df/2,ficont+df/2,wi)-TF('FS',FSfe-df/2,ficont+df/2,wi))\
    #             - (TF('FS',FSfe+df/2,ficont-df/2,wi)-TF('FS',FSfe-df/2,ficont-df/2,wi)) )/df**2
    # dvidviFi = ( TF('FS',FSfe,ficont+df,wi) - 2*Fi + TF('FS',FSfe,ficont-df,wi) )/df**2
    # dvedviFi = ( (TF('FS',FSfe+df/2,ficont+df/2,wi)-TF('FS',FSfe+df/2,ficont-df/2,wi))\
    #             - (TF('FS',FSfe-df/2,ficont+df/2,wi)-TF('FS',FSfe-df/2,ficont-df/2,wi)) )/df**2
    
    
    #-------INTEGRATION--------------------------------------------------------------------------

    #-first order EULER
    fecont += dt/T*( (Fe-fecont) )
    ficont += dt/T*( (Fi-ficont) )

    #-first order HEUN
    # fecont += dt/T*(Fe-fecont)
    # fecont = fecontold + dt/T/2*(Fe-fecontold + TF('RS',RSfe,ficont,we)-fecont)
    # ficont += dt/T*(Fi-ficont)
    # ficont = ficontold + dt/T/2*(Fi-ficontold + TF('FS',FSfe,ficont,wi)-ficont)

    #-second order EULER
    # fecont += dt/T*( (Fe-fecont) + (cee*dvedveFe+cei*dvedviFe+cii*dvidviFe+cei*dvidveFe)/2 )
    # ficont += dt/T*( (Fi-ficont) + (cee*dvedveFi+cei*dvedviFi+cii*dvidviFi+cei*dvidveFi)/2 )

    #-second order HEUN
    # fecont += dt/T*( (Fe-fecont) + (cee*dvedveFe+cei*dvedviFe+cii*dvidviFe+cei*dvidveFe)/2 )
    # fecont = fecontold + dt/T/2*( (Fe-fecontold) + (TF('RS',RSfe,ficont,we)-fecont) + (cee*dvedveFe+cei*dvedviFe+cii*dvidviFe+cei*dvidveFe) )
    # ficont += dt/T*( (Fi-ficont) + (cee*dvedveFi+cei*dvedviFi+cii*dvidviFi+cei*dvidveFi)/2 )
    # ficton = ficontold + dt/T/2*( (Fi-ficontold) + (TF('FS',FSfe,ficont,wi)-ficont) + (cee*dvedveFi+cei*dvedviFi+cii*dvidviFi+cei*dvidveFi) )


    #-adaptation EULER
    we += dt*MFw('RS',we,fecontold,ficontold)
    wi += dt*MFw('FS',wi,0,ficontold)

    #-adaptation HEUN
    # we += dt*MFw('RS',we,fecontold,0)
    # we = weold + dt/2*( MFw('RS',weold,fecontold,0) + MFw('RS',we,fecontold,0) )
    # wi += dt*MFw('FS',wi,FSfe,ficontold)
    # wi = wiold + dt/2*( MFw('FS',wiold,FSfe,ficontold) + MFw('FS',wi,FSfe,ficontold) )

    if fecont<1e-9: fecont=1e-9
    if ficont<1e-9: ficont=1e-9
    if fecont>175: fecont=175
    if ficont>175: ficont=175

    LSfe.append(float(fecont))
    LSfi.append(float(ficont))
    LSwe.append(float(we))
    LSwi.append(float(wi))


    #-covariances EULER
    # cee += dt/T*( Fe*(1/T-Fe)/8000 + (Fe-fecontold)**2 + 2*cee*dveFe + 2*ceiold*dviFe - 2*cee)
    # cei += dt/T*( (Fe-fecontold)*(Fi-ficontold) + cei*dveFe + ceeold*dveFi + ciiold*dviFe + cei*dviFi - 2*cei)
    # cii += dt/T*( Fi*(1/T-Fi)/2000 + (Fi-ficontold)**2 + 2*cii*dviFi + 2*ceiold*dveFi - 2*cii)

    #-covariances HEUN
    # cee += dt/T*( Fe*(1/T-Fe)/500 + (Fe-fecontold)**2 + 2*cee*dveFe + 2*cei*dviFe - 2*cee)
    # cee = ceeold + dt/T*( Fe*(1/T-Fe)/500 + (Fe-fecontold)**2 + ceeold*dveFe + 2*cei*dviFe - ceeold + cee*dveFe - cee)
    # cei += dt/T*( (Fe-fecontold)*(Fi-ficontold) + cee*dveFi + cei*dveFe + cei*dviFi + cii*dviFe - 2*cei)
    # cei = ceiold + dt/T*( (Fe-fecontold)*(Fi-ficontold) + cee*dveFi + ceiold*dveFe/2 + ceiold*dviFi/2 + cii*dviFe - ceiold + cei*dveFe/2 + cei*dviFi/2 - cei)
    # cii += dt/T*( Fi*(1/T-Fi)/500 + (Fi-ficontold)**2 + 2*cei*dveFi + 2*cii*dviFi - 2*cii)
    # cii = ciiold + dt/T*( Fi*(1/T-Fi)/500 + (Fi-ficontold)**2 + 2*cei*dveFi + ciiold*dviFi - ciiold + cii*dviFi - cii)

    # if cee<1e-9: cee=1e-9
    # if cii<1e-9: cii=1e-9
    # if cei<1e-9: cei=1e-9
    # LScee.append(np.sqrt(cee))
    # LScii.append(np.sqrt(cii))

    #-test
    # test.append(dvedveFe)
    # test2.append(dvedviFe)
    # test2.append(2*cii*dviFi + 2*ceiold*dveFi)


#-end of loop

LSfe=np.array(LSfe)
LSfi=np.array(LSfi)
LSwe=np.array(LSwe)
LSwi=np.array(LSwi)
# LScee=np.array(LScee)
# LScii=np.array(LScii)


#----SAVING--------------------------------------------

np.save('data\\MF_out', np.vstack((LSfe,LSfi)))
# np.save('data\\MF_out_cov', np.vstack((LScee,LScii)))

# np.savetxt('test.txt',test)


#------PLOTTING-----------------------------------------------

#-testplots
# plt.plot(test)
# plt.plot(test2)
# plt.show()
# plt.plot(LScee, 'b')
# plt.plot(LScii, 'r')
# plt.show()

#-main plot
fig = plt.figure()
fig.subplots_adjust(hspace=0.001)
gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
ax3=fig.add_subplot(gs[0])
ax2=fig.add_subplot(gs[1],sharex=ax3)
ax1=ax3.twinx()

ax3.set_axis_off()
ax1.tick_params(labelright=False,labelbottom=False,labelleft=True,labeltop=False,which='both',
                left=True,right=True,bottom=False, top=False)
ax2.tick_params(which='both',right=True,top=True,grid_alpha=0.3)
ax2.tick_params(axis='y', labelsize=8, size=2,grid_alpha=0)

ax1.set_xlim(0,tfinal)
maxpoint=max(np.concatenate([LSfe,LSfi]))
ax1.set_ylim(-maxpoint/10,maxpoint+maxpoint/5)

ax1.plot(t, LSfe, c='b', label=r'$\nu_{\mathrm{RS}}$')
# ax1.fill_between(t, LSfe-LScee, LSfe+LScee, color='b', label=r'$\sigma_{\mathrm{RS}}$', alpha=0.2)
ax1.plot(t, LSfi, c='r', label=r'$\nu_{\mathrm{FS}}$')
# ax1.fill_between(t, LSfi-LScii, LSfi+LScii, color='r', label=r'$\sigma_{\mathrm{FS}}$', alpha=0.2)
ax1.plot(t,external_input, c='black', label=r'$P_C$')
ax1.plot(t,stim, c='black', ls='--', label=r'$P_S$')

ax2.grid()

ax2.plot(t, LSwe*1e12, c='b', label=r'$\omega_{\mathrm{RS}}$')
ax2.plot(t, LSwi*1e12, c='r', label=r'$\omega_{\mathrm{FS}}$')


ax1.yaxis.set_label_position('left')
ax1.set_ylabel(r'frequency $\nu$ [Hz]',fontsize=12)

ax2.set_xlabel(r'time $t$ [s]',fontsize=12)
ax2.set_ylabel(r'adaptation $\omega$ [pA]',fontsize=10,position=(0,0.5))

leg1 = ax1.legend(bbox_to_anchor=(1.205, 1.0), loc=1, borderaxespad=0.)
leg2 = ax2.legend(bbox_to_anchor=(1.215, 1.0), loc=1, borderaxespad=0.)
ax1.add_artist(leg1)

plt.savefig('gfx\\MF_PLOT.png', dpi=200, bbox_inches='tight')

#------END------------------------------------------------------------------

