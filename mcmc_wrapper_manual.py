import numpy as np
import json
import run_mcmc_fitter_candy_manual
import os

import datetime
start_time = datetime.datetime.now().time()

# Create output .txt file and set up header
#outtxt = 'MCMC_results_500_fullprofile_star_10_pilot_J1534+5015a.txt'
outtxt = 'new_ref_fits/J1212+2826a_newfit.txt'
#outtxt = 'new_ref_fits/test.txt'
with open(outtxt,'w') as file:
    file.write("Name"+"\t"+"Line"+"\t"+"lam_red"+"\t"+"lam_red+quant"+"\t"+"lam_red-quant"+"\t"+"LogN"+"\t"+"LogN+quant"+"\t"+"LogN-quant"+"\t"+"bD"+"\t"+"bD+quant"+"\t"+"bD-quant"+"\n")


"""
# THIS IS FOR MANUAL RUNS


# BHB = 'J1149+2828a' # test manual runs with BHB that has run successfully before
# cmps = ['z-0.00017_CaII',                         'z0.00034_CaII',                        'z-0.00019_NaI']
# zfit = [-0.00017121871450343032,                  0.00034,                                -0.00018155618609208034]
# vlim = [[-20.238017125404212,20.238017125442067], [-64.58941635775038,64.58941635774099], [-17.223844362100035,17.223844362048265]]
# Nfit = [12.180042112128683,                       12.53,                                  11.24573875510486]
# bfit = [8.5555694740381,                          23.6,                                   7.125715467130997]

# BHB = 'J1341+2823a'
# #cmps = ['z-0.00014_NaI',                            'z-0.00015_CaII',                           'z-0.00050_CaII'] # all components
# #cmps = ['z-0.00014_NaI',                            'z-0.00015_CaII'] # no star
# cmps = ['z-0.00050_CaII'] # star
# zfit = [-0.000145,                                  -0.00014,                                   -0.0004833061927935164]
# vlim = [[-14.022826044633748, 14.0228260446253],    [-15.035705398733015,15.03570539880534],    [-58.42844185259096,58.42844185255476]]
# Nfit = [11.6,                                       12.0,                                       12.228458261670713]
# bfit = [3.0,                                        5.0,                                        25.91588939366904]

# BHB = 'J1527+4027a'
# #cmps = ['z-0.00006_CaII',                           'z-0.00007_NaI',                            'z-0.00024_CaII',                           'z-0.00029_CaII',                       'z-0.00040_CaII'] # all components
# #cmps = ['z-0.00006_CaII',                           'z-0.00007_NaI',                            'z-0.00024_CaII',                           'z-0.00029_CaII'] # no star
# cmps = ['z-0.00040_CaII'] # star
# zfit = [-7e-05,                                     -6.5e-05,                                   -0.00023,                                   -0.000295,                               -0.00037102453426500086]
# vlim = [[-20.243144478291782,20.24314447854732],    [-12.575286721372844,12.575286721419715],   [-7.328099639747751,7.328099639739855],     [-8.685155128465526,8.6851551286067],    [-24.537144822167164,24.53714482220668]]
# Nfit = [11.85,                                      11.55,                                      11.5,                                       11.5,                                    12.401660616552567]
# bfit = [5.0,                                        4.0,                                        10.0,                                       5.0,                                     10.0]

# BHB = 'J1534+5015a'
# #cmps = ['z-0.00004_CaII',                           'z-0.00005_NaI',                            'z-0.00015_CaII',                           'z-0.00016_NaI',                            'z-0.00023_CaII',                            'z-0.00046_CaII'] # all components
# #cmps = ['z-0.00004_CaII',                           'z-0.00005_NaI',                            'z-0.00015_CaII',                           'z-0.00016_NaI',                            'z-0.00023_CaII'] # no star
# cmps = ['z-0.00046_CaII'] # star
# zfit = [-6e-05,                                     -6e-05,                                     -0.00015,                                   -0.00016,                                   -0.00023,                                    -0.00043865416593692584]
# vlim = [[-20.15865645708421,20.158656457133205],    [-19.84281142233846,19.842811422175593],    [-10.960351715762796,10.960351715754467],   [-16.65131028448983,16.651310284456784],    [-10.960351715740423,10.96035171575381],     [-53.95865460068079,53.958654600650966]]
# Nfit = [12.1,                                       11.7,                                       12.77,                                      11.8,                                       11.9,                                        12.588200324730098]
# bfit = [10.0,                                       4.0,                                        4.0,                                        5.0,                                        5.0,                                         26.954616484368174]

BHB = 'J1223+0002a'
# test tweaking parameters for one component only
#cmps = ['z-0.00046_CaII'] # star
cmps = ['z1_0.000145_CaII', 'z2_0.000145_CaII', 'z3_0.000145_CaII', 'z4_0.000145_CaII', 'z5_0.000145_CaII', 'z6_0.000145_CaII', 'z7_0.000145_CaII'] 
zfit = [0.000145,           0.000145,           0.000145,           0.000145,           0.000145,           0.000145,           0.000145]
vlim = [[-10.0,10.0],    [-10.0,10.0],    [-10.0,10.0],   [-10.0,10.0],    [-10.0,10.0],     [-10.0,10.0],     [-10.0,10.0]]
Nfit = [11.5,               11.4,               11.2,               11.6,               11.6,               11.7,               11.8]
bfit = [1.,                 3.,                 2.,                 4.,                 5.,                 6.,                 9.]

# BHB = 'J1212+2826a'
# # tweaking parameters for one component only
# cmps = ['z-0.00005_NaI',     'z_NaI_MW'] 
# zfit = [-0.00003,       -0.00003]
# vlim = [[-17.0,17.0],   [-17.0,17.0]]
# Nfit = [11.60,          11.60]
# bfit = [6.0,            6.0]



## All BHBs
inputfits = '~/Dropbox/BHB_abs/allreduced/' + BHB + '_ltnorm_cor.fits'
comp_dict = 0
components = []


for i, cmp in enumerate(cmps):
    components.append(str(cmp))
    json = {'zfit':zfit[i], 'vlim':vlim[i], 'Nfit':Nfit[i], 'bfit':bfit[i]}
    if str(cmp)[-4] == "_": #This should be the case if the line is NaI
        species = str(cmp)[-3:]
        z_component = str(cmp)[1:-4]
    elif str(cmp)[-5] == "_": #This should be the case if the line is CaII
        species = str(cmp)[-4:]
        z_component = str(cmp)[1:-5]
    else:
        print "Error retrieving species and component"
        os.system("Error reading file string")
        break
    print "Now fitting " + BHB + ": " + str(cmp)
    print "z = " + z_component
    print "Species: " + species
    run_mcmc_fitter_candy_manual.run_all(inputfits,json,outtxt,z_component,species,BHB)
print "Time: ", datetime.datetime.now().time()
alert_command = "say '"+BHB+"'"
os.system(alert_command)
"""

#"""
### THIS IS FOR AUTOMATED RUNS
targnames = np.loadtxt('targnames_keck.txt',dtype='str',comments='#')
#targnames = np.array(['J1527+4027a']) ####################
print targnames
for BHB in targnames:
    ## All BHBs
    inputfits = '~/Dropbox/BHB_abs/allreduced/' + BHB + '_ltnorm_cor.fits'
    inputjson = '../../BHB_abs/allreduced/' + BHB + '_new.json'
    #inputjson = 'test.json'

    with open(inputjson) as json_file:
        linefits=json.load(json_file)
    comp_dict = 0
    components = []
    for i, cmp in enumerate(linefits["cmps"]):
        #XXXXXX*************************!!!!!!CURRENTLY SET TO RUN ON SUBSET OF LINES!!!!!!!***********************
        if(linefits["cmps"][str(cmp)]["comment"] != "blah"):
        #if(linefits["cmps"][str(cmp)]["comment"] == "star"):
        #if(linefits["cmps"][str(cmp)]["comment"] != "star"):
        #if((linefits["cmps"][str(cmp)]["Comment"] == "MW") and (linefits["cmps"][str(cmp)]["Comment"] != "star")):
            print ' '
            components.append(str(cmp))
            comp_dict=linefits["cmps"][str(cmp)]
            #print str(cmp)[8]
            #print str(cmp)[9]
            if str(cmp)[-4] == "_": #This should be the case if the line is NaI
                species = str(cmp)[-3:]
                z_component = str(cmp)[1:-4]
            elif str(cmp)[-5] == "_": #This should be the case if the line is CaII
                species = str(cmp)[-4:]
                z_component = str(cmp)[1:-5]
            else:
                print "Error retrieving species and component from JSON string"
                os.system("Error reading file string")
                break
            print "----------------------------------- Now fitting " + BHB + ": " + str(cmp) + " (" + str(i+1) + " of " + str(len(linefits["cmps"])) + ") -----------------------------------"
            print "z = " + z_component
            print "Species: " + species
            run_mcmc_fitter_candy_manual.run_all(inputfits,inputjson,outtxt,z_component,species,BHB)
        print "Time: ", datetime.datetime.now().time()
    alert_command = "say '" + BHB + "'"
    os.system(alert_command)
#"""


print "Start time: ", start_time
print "End time  : ", datetime.datetime.now().time()
os.system('afplay /System/Library/Sounds/Glass.aiff')
os.system('afplay /System/Library/Sounds/Glass.aiff')
os.system('afplay /System/Library/Sounds/Glass.aiff')
#os.system('afplay /System/Library/Sounds/Glass.aiff')
#os.system('afplay /System/Library/Sounds/Glass.aiff')
