import numpy as np
import json
import run_mcmc_fitter_candy
import os

targnames = np.loadtxt('targnames_keck.txt',dtype='str',comments='#')

# Create output .txt file and set up header
outtxt = 'MCMC_results_absorption_stack_output.txt'
with open(outtxt,'w') as file:
    file.write("Name"+"\t"+"Line"+"\t"+"lam_red"+"\t"+"lam_red+quant"+"\t"+"lam_red-quant"+"\t"+"LogN"+"\t"+"LogN+quant"+"\t"+"LogN-quant"+"\t"+"bD"+"\t"+"bD+quant"+"\t"+"bD-quant"+"\n")

for BHB in targnames:
    ## 2016 run
    #inputfits = '~/Dropbox/BHB_abs/HIRES-March2016/Rawdata/2016mar30/NormSpec/' + BHB + '_ltnorm_cor.fits'
    #inputjson = '../../BHB_abs/HIRES-March2016/Rawdata/2016mar30/NormSpec/' + BHB + '.json'
    ## feb20 run
    #inputfits = '~/Dropbox/BHB_abs/Keck2017A/DATA/2017feb20/NormSpec/' + BHB + '_ltnorm_cor.fits'
    #inputjson = '../../BHB_abs/Keck2017A/DATA/2017feb20/NormSpec/' + BHB + '.json'
    ## feb21 run
    #inputfits = '~/Dropbox/BHB_abs/Keck2017A/DATA/2017feb21+feb20arcs/NormSpec/' + BHB + '_ltnorm_cor.fits'
    #inputjson = '../../BHB_abs/Keck2017A/DATA/2017feb21+feb20arcs/NormSpec/' + BHB + '.json'
    ## pilot BHBs
    #inputfits = '~/Dropbox/BHB_abs/HIRES_BHB/pilotredux/' + BHB + '_ltnorm_cor.fits'
    #inputjson = '../../BHB_abs/HIRES_BHB/pilotredux/' + BHB + '.json'
    ## All BHBs
    inputfits = '~/Dropbox/BHB_abs/allreduced/' + BHB + '_ltnorm_cor.fits'
    inputjson = '../../BHB_abs/allreduced/' + BHB + '.json'

    with open(inputjson) as json_file:
        linefits=json.load(json_file)
    comp_dict = 0
    components = []
    for cmp in linefits["cmps"]:
        #XXXXXX*************************!!!!!!CURRENTLY NOT SET TO RUN ON MW LINES ONLY!!!!!!!***********************
        if(linefits["cmps"][str(cmp)]["comment"] != "blah"):
        #if(linefits["cmps"][str(cmp)]["comment"] != "star"):
        #if((linefits["cmps"][str(cmp)]["Comment"] == "MW") and (linefits["cmps"][str(cmp)]["Comment"] != "star")):
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
            print "Now fitting " + BHB + ": " + str(cmp)
            print "z = " + z_component
            print "Species: " + species
            run_mcmc_fitter_candy.run_all(inputfits,inputjson,outtxt,z_component,species,BHB)
    alert_command = "say '"+BHB+"'"
    os.system(alert_command)
