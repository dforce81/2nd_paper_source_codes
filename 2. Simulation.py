# 2nd paper 2. Monte Carlo simulation 
# forecasted solar data : real data
# real data to simulate : sampling from real data


# Call the AMPL-Phthon API to Python    
from amplpy import AMPL, DataFrame, Environment

# Call some packages  
import numpy as np
import pandas as pd


# Determine the path of the directory where AMPL is installed.
ampl = AMPL(Environment('path of AMPL'))


# Call energy scheduling model file and data file from the path of the directory where AMPL files are located
ampl.read('Energy_Scheduling.mod')
ampl.readData('Energy_Scheduling.dat')
   

# Determine the solver from AMPL
ampl.setOption('solver','C:/Users/David_Cho/Desktop/ampl_mswin64/cplex')   
 
  
# Define variables to calculate yearly data results
global d_loss_sim 
d_loss_sim = 0

decide_running = 0    

yearly_energy_generated_forecasted = {}
yearly_energy_generated_actual = {}
    
yearly_energy_appliance_total_forecasted = {}
yearly_energy_appliance_total_actual = {}

yearly_energy_appliance_pv_forecasted = {}
yearly_energy_appliance_pv_actual = {}
               
yearly_energy_appliance_battery_forecasted = {}
yearly_energy_appliance_battery_actual = {}

yearly_energy_charged_forecasted = {}
yearly_energy_charged_actual = {}   

yearly_energy_loss_pv_forecasted= {}
yearly_energy_loss_pv_actual = {}

yearly_energy_loss_conv_forecasted = {}
yearly_energy_loss_conv_actual = {}

yearly_energy_loss_eapp_bat_forecasted = {}
yearly_energy_loss_eapp_bat_actual = {}    

yearly_energy_loss_chr_forecasted = {}
yearly_energy_loss_chr_actual = {}   

yearly_energy_loss_total_forecasted = {}
yearly_energy_loss_total_actual = {}

yearly_appliance_operation_hour_forecasted = {}
yearly_appliance_operation_hour_actual = {}

yearly_energy_demand_forecasted = {}
yearly_energy_demand_actual = {}
yearly_energy_demand_total_actual = {}

annual_demand_satisfaction = {}


# Define variables to calculate daily data results
daily_energy_generated_forecasted = {}
daily_energy_generated_actual = {}

daily_energy_demand_forecasted = {}
daily_energy_demand_actual = {}

daily_energy_appliance_total_forecasted = {}  
daily_energy_appliance_total_actual = {}  

daily_energy_appliance_pv_forecasted = {}
daily_energy_appliance_pv_actual = {}

daily_energy_appliance_battery_forecasted = {}
daily_energy_appliance_battery_actual = {}

daily_energy_charged_forecasted = {}
daily_energy_charged_actual = {}

daily_energy_loss_pv_forecasted = {} 
daily_energy_loss_pv_actual = {} 

daily_energy_loss_conv_forecasted = {} 
daily_energy_loss_conv_actual = {} 

daily_energy_loss_eapp_bat_forecasted = {} 
daily_energy_loss_eapp_bat_actual = {} 

daily_energy_loss_chr_forecasted = {}
daily_energy_loss_chr_actual = {}
   
daily_energy_loss_total_forecasted = {} 
daily_energy_loss_total_actual = {} 
daily_energy_loss_total_actual_2 = {}

daily_appliance_operation_hour_forecasted = {}
daily_appliance_operation_hour_actual = {}   

avg_hourly_soc = {}

hourly_soc = {}

soc_last_period = {}

total_op_time_applaince_weekday = {}
total_op_time_applaince_weekend = {}
total_op_time_applaince_holiday = {}


# Define variables to create Weekly operation for washer and dryer
washer = {}
dryer = {}

daily_opt_time_Washer = {}
daily_opt_time_Dryer = {}

weekly_opt_time_Washer = {}
weekly_opt_time_Dryer = {}

yearly_opt_time_Washer = {}
yearly_opt_time_Dryer = {}


# Call parameters to initialize pririties and operation time of washer and dryer drom AMPL dat file
priority_Washer  = ampl.param['W_SCENARIO'][ampl.getValue('DATE[1,1]'),1] # initial value
priority_Dryer = ampl.param['W_SCENARIO'][ampl.getValue('DATE[1,1]'),2] # initial value

opt_time_Washer = ampl.param['T_SCENARIO'][ampl.getValue('DATE[1,1]'),1] # initial value
opt_time_Dryer = ampl.param['T_SCENARIO'][ampl.getValue('DATE[1,1]'),2] # initial value


#Assign x_pv, x-bat to AMPL to run
ampl.param['RPV'] = x_pv / ampl.getValue('APV')
ampl.param['CBAT'] = x_bat

ampl.param['RMCH'] = x_bat * 0.51
ampl.param['RMDC'] = x_bat * 0.51


# Pre-check the values of x_pv, x_bat
# if the value of x_pv and x_bat that are called from Integer_N-M.py do not satisfy model running condition, it assigs a large value to a solution of x_pv and x_bat. 
if x_pv <= 0 :
    if x_bat <= 0 :      
        print("exit")    
        d_loss_sim = 1000000 + 1000000
        print("Failed : PV size : ",  x_pv, "and Battery capacity :", x_bat, "cannot satisfy model running condition.")    
        print("Assign the higest d_loss cost", d_loss_sim ,"for the point (", x_pv, ",", x_bat, ")")   


    else :
        print("exit")    
        d_loss_sim = 1000000 + 1000000/ x_bat
        print("Failed :PV size : ",  x_pv, "and Battery capacity :", x_bat, "cannot satisfy model running condition.")    
        print("Assign the higest d_loss cost", d_loss_sim ,"for the point (", x_pv, ",", x_bat, ")")    
        

else :
    if x_bat <= 0 :      
        print("exit")
        d_loss_sim = 1000000 / x_pv + 1000000
        print("Failed :PV size : ",  x_pv, "and Battery capacity :", x_bat, "cannot satisfy model running condition.")    
        print("Assign the higest d_loss cost", d_loss_sim ,"for the point (", x_pv, ",", x_bat, ")")    


# Passed : Pre-checking-1 if (x_pv and/or x_bat) < 0        
    else :
        print(" ")
    
        ## calling parameter values from AMPL .dat to speed up the computational time
        
        # vetorize iteration to speed up the model running
        week_iter = np.arange(52, dtype =np.float64)+1
        day_iter = np.arange(7, dtype =np.float64)+1
        time_iter = np.arange(24, dtype =np.float64)+1
        appliance_iter = np.arange(25, dtype =np.float64)+1
        pv_iter = np.arange(1, dtype =np.float64)+1
        bat_iter = np.arange(1, dtype =np.float64)+1
    
        # Assign parameters to python variable from AMPL files to speed up the computational time
        ampl_sisc = ampl.getValue('SISC')
        ampl_rpv = ampl.getValue('RPV')
        ampl_apv = ampl.getValue('APV')
        ampl_rdch = ampl.getValue('RDCH')
        ampl_rmch = ampl.getValue('RMCH')
        ampl_rchr = ampl.getValue('RCHR')  
        ampl_smax = ampl.getValue('SMAX')
        ampl_smin = ampl.getValue('SMIN')
        ampl_rinv = ampl.getValue('RINV')
        ampl_send = ampl.getValue('SEND')
        ampl_rmdc = ampl.getValue('RMDC')
           
        # Call parameter data in excel file to dataframe : .dat        
        xls =pd.ExcelFile ('model_parameters.xlsx') 
        df_dtype = pd.read_excel(xls, sheet_name='dtype')
        df_solar = pd.read_excel(xls, sheet_name='solar')
        df_p = pd.read_excel(xls, sheet_name='p')
        df_t = pd.read_excel(xls, sheet_name='t')
        df_eapp = pd.read_excel(xls, sheet_name='eapp')
        df_em = pd.read_excel(xls, sheet_name='em')
        df_w = pd.read_excel(xls, sheet_name='w')
        
        # set index in the dataframe
        df_dtype.set_index(['w', 'd'], inplace=True) 
        df_solar.set_index(['w', 'd', 't'], inplace=True) 
        df_p.set_index(['date_type', 'a', 't'], inplace=True) 
        df_t.set_index(['date_type', 'a'], inplace=True) 
        df_eapp.set_index(['date_type', 'a'], inplace=True) 
        df_em.set_index(['date_type', 't'], inplace=True) 
        df_w.set_index(['date_type', 'a'], inplace=True) 
    
        # convert dataframe into dictionary
        solar = df_solar.to_dict()
        d_type = df_dtype.to_dict()
        em = df_em.to_dict()
        param_p = df_p.to_dict()
        param_eapp = df_eapp.to_dict()
        param_t = df_t.to_dict()        
        param_w = df_w.to_dict()

        # Define variables to pre-check       
        epv = {}
        ebat = {}
        ebat_actual = {}
        epv_actual = {}
        esystem = {}


        # initialize yearly results variables for next PV and battery combination                        
        yearly_energy_generated_forecasted = 0
        yearly_energy_generated_actual = 0
    
        yearly_energy_appliance_total_forecasted = 0
        yearly_energy_appliance_total_actual = 0

        yearly_energy_appliance_pv_forecasted = 0
        yearly_energy_appliance_pv_actual = 0
               
        yearly_energy_appliance_battery_forecasted = 0
        yearly_energy_appliance_battery_actual = 0

        yearly_energy_charged_forecasted = 0
        yearly_energy_charged_actual = 0   

        yearly_energy_loss_pv_forecasted = 0
        yearly_energy_loss_pv_actual  = 0
        
        yearly_energy_loss_conv_forecasted  = 0
        yearly_energy_loss_conv_actual  = 0

        yearly_energy_loss_eapp_bat_forecasted  = 0
        yearly_energy_loss_eapp_bat_actual  = 0

        yearly_energy_loss_chr_forecasted  = 0
        yearly_energy_loss_chr_actual = 0

        yearly_energy_loss_total_forecasted = 0
        yearly_energy_loss_total_actual  = 0

        yearly_appliance_operation_hour_forecasted = 0
        yearly_appliance_operation_hour_actual = 0

        yearly_energy_demand_forecasted  = 0
        yearly_energy_demand_actual  = 0
        yearly_energy_demand_total_actual  = 0

        annual_demand_satisfaction  = 0
 
        yearly_opt_time_Washer  = 0
        yearly_opt_time_Dryer = 0
        

        ## Pre-check-2 if x_pv and x_bat meets minimum demand - energy margin(0.5kWh) during particular period based on the type of the day 
        # Changes in week w from 1 to 52 : 52 weeks in a year           
        for w in week_iter :
            # Changes in day d from 1 to 7 : 7 days in a week
            for d in day_iter :    
                # Changes in time t from 1 to 24 during a day
                for t in time_iter :                   

                    ampl.param['EM'][t] = em['em'][d_type['date_type'][w,d],t]  # Changes in parameter EM[t](energy margin) based on day d                                                             
                    epv[t] =  ampl_rpv *  solar['s_forecasted'] [w,d,t] * ampl_apv # horuly energy generated by PV array
                    ebat[0] = ampl_sisc * x_bat # initialize amount of energy in the battery
                    ebat_actual[0] = ampl_sisc * x_bat # initialize actual amount of energy in the battery
                    ebat[t] = min(ebat_actual[t-1] * (1 - ampl_rdch) + min(ampl_rmch, epv[t] * ampl_rchr), x_bat * ampl_smax)                                    
                    epv_actual[t] = max(ebat_actual[t-1] * (1 - ampl_rdch) + epv[t] - x_bat * ampl_smax, 0) / ampl_rchr

                    # hourly available amount of energy in the system when there is energy margin
                    esystem[t] = ebat[t] + epv_actual[t] - em['em'][d_type['date_type'][w,d],t] / ampl_rinv
                   
                    # available amount of energy in the battery when there is energy margin
                    # epv is used to energy margin before energy in a battery
                    ebat_actual[t] = min(esystem[t], x_bat * ampl_smax)  
                    
                    # Pre-check if SOC of x_bat satisfy minimum hourly SOC (=5%)
                    if ebat_actual[t] / x_bat < ampl_smin  or  esystem[t] < 0 :
                        
                        decide_running = decide_running + 1
                        
                        # assign the value of esystem after t and break 
                        for t_sub in range(int(t)+1, 24+1) : # t is numpy.float64 but we need to change it into int when we use range()
                            esystem[t_sub] = float(0)
                            
                        break   
               
                # Pre-check if amount of energy at the last period of the day >= 30% of SOC 
                if  esystem[24] - em['em'][d_type['date_type'][w,d],24] / ampl_rinv < ampl_send * x_bat :
                    
                    decide_running = decide_running + 1

                    break                    

            if decide_running > 0 :
                
                break    

        # If precheck do not satisfy minimum energy demand, the energy scheduling model will not run. Assign a large amount of d_loss to (x_pv, x_bat) combination.
        if decide_running > 0 :
            
            d_loss_sim = 1000000 / x_pv + 1000000 / x_bat

            print(" ")                                       
            print("Failed : Pre-check if x_pv and x_bat satisfy hourly energy margin(=0.5kWh)")
            print("Failed : Pre-check if amount of energy at the last period of the day >= 30% of SOC")
            print(" ")           
            print("PV size : ",  x_pv, "and Battery capacity :", x_bat, "cannot satisfy model running condition.")    
            print("Assign a large d_loss cost", d_loss_sim ,"for the point (", x_pv, ",", x_bat, ")")  

        # if pre-check satisfies minimum energy demand, the energy scheduling model will run with the simulation model.
        else :        
            print("Energy scheduling model and simulation model are going to running")



################################## Run the energy scheduling optimization model for a year (day 1 on week 1 to day 364 on week 52) ######################################################################

            # Initialize Avg. hourly SOC for the anlayis purpose
            for t in time_iter :

                hourly_soc[t] = 0
                avg_hourly_soc[t] = 0

            # Initialize total operation tiem of each applaince for the anlayis purpose              
            for a in appliance_iter :
                
                total_op_time_applaince_weekday[a] = 0                  
                total_op_time_applaince_weekend[a] = 0
                total_op_time_applaince_holiday[a] = 0      
            
            # Changes in week w from 1 to 52 : 52 weeks in a year           
            for w in week_iter :
              
                weekly_opt_time_Washer[w] = 0 # initialize cumulative operation time of washer every first day of new week  # dynamic profiling for washing machine 
                weekly_opt_time_Dryer[w] = 0 # initialize cumulative operation time of dryer every first day of new week  # dynamic profiling for dryer

                # Changes in day d from 1 to 7 : 7 days in a week
                for d in day_iter :

                    ampl.param['SISC'] = ampl_sisc
                    
                    # Changes in time t from 1 to 24 during a day
                    for t in time_iter :
                        
                        ampl.param['SPV'][t] = solar['s_forecasted'] [w,d,t]  # Changes in parameter SPV(hourly forecasted solar irradiance) based on time t on day d of week w
                                   
                    # Changes in parameters of appliances based on the type of day d                   
                    for a in appliance_iter :
                        
                        for t in time_iter :     
                            
                            ampl.param['P'][a,t] = param_p['p'][d_type['date_type'][w,d],a,t]  # Changes in parameter P[a,t](preferred operation time for appliance) based on day d
                            ampl.param['EM'][t] = em['em'][d_type['date_type'][w,d],t]  # Changes in parameter EM[t](energy margin) based on day d
                          
                        ampl.param['EAPP'][a] = param_eapp['eapp'][d_type['date_type'][w,d],a]  # Changes in parameter EAPP[a](required energy by appliance) based on day d
                        ampl.param['T'][a] = param_t['t'][d_type['date_type'][w,d],a]  # Changes in parameter T[a](appliance time to operate) based on day d
                        ampl.param['W'][a] = param_w['w'][d_type['date_type'][w,d],a]  # Changes in parameter W[a](priority of appliance) based on day d
                  
                    ampl.param['W'][1] = priority_Washer   # weekly required operation : update priority of washer
                    ampl.param['W'][2] = priority_Dryer   # weekly required operation : update priority of dryer
                    ampl.param['T'][1] = opt_time_Washer  # weekly required operation : update operation time of washer
                    ampl.param['T'][2] = opt_time_Dryer  # weekly required operation : update operation time of dryer
                          
                    print("")                
                    print("Type of day {d} of week {w} : ".format(d=d, w=w), ampl.getValue('DATE[{w},{d}]'.format(w=w, d=d)))  
                    print("PV size :", ampl_rpv*ampl_apv, "kW", "/", "Battery capacity :", x_bat, "kWh")                            
                    print("SISC for day {d} of week {w} : ".format(d=d, w=w), ampl.getValue('SISC'))                                                 
   
                    ampl.solve()  # solving the model : obtain pre-schedules of applainces operation based on the forecasted solar irradiance data

                    ampl.display('sbat[24]*CBAT;') # pre-schedules of applainces operation

                    daily_energy_generated_forecasted [w,d] = ampl.getValue ('sum{t in Time} EPV[t];')
                    yearly_energy_generated_forecasted = yearly_energy_generated_forecasted  + daily_energy_generated_forecasted [w,d]  # yearly energy generated by pv system from AMPL model.

                    daily_energy_demand_forecasted [w,d] = ampl.getValue ('sum{a in Appliance} EAPP[a] * T[a] + sum{t in Time} EM[t];') # yearly energy demand -> since it doesn't consider operation of weekly appliances, it might be greater than actual.      
                    yearly_energy_demand_forecasted  = yearly_energy_demand_forecasted  + daily_energy_demand_forecasted [w,d] # yearly energy demand -> since it doesn't consider operation of weekly appliances, it might be greater than actual.      
 
                    daily_energy_appliance_total_forecasted [w,d] = ampl.getValue ('sum{t in Time, a in Appliance} EAPP[a] * xapp_state[a,t] + sum{t in Time} EM[t];')
                    yearly_energy_appliance_total_forecasted  = yearly_energy_appliance_total_forecasted  + daily_energy_appliance_total_forecasted [w,d] # yearly energy used by appliances -> since it doesn't consider operation of weekly appliances, it might be greater than actual.      

                    daily_energy_appliance_pv_forecasted [w,d] = ampl.getValue('sum{t in Time} eapp_pv[t];')
                    yearly_energy_appliance_pv_forecasted  = yearly_energy_appliance_pv_forecasted  + daily_energy_appliance_pv_forecasted [w,d]  # yearly energy used by appliance from pv                

                    daily_energy_appliance_battery_forecasted [w,d] = ampl.getValue ('sum{t in Time} eapp_bat[t];')  
                    yearly_energy_appliance_battery_forecasted  = yearly_energy_appliance_battery_forecasted  + daily_energy_appliance_battery_forecasted [w,d]  # yearly energy used by appliance from battery

                    daily_energy_charged_forecasted [w,d] =  ampl.getValue('sum{t in Time} echr[t];') 
                    yearly_energy_charged_forecasted  = yearly_energy_charged_forecasted  + daily_energy_charged_forecasted [w,d] # yearly energy charged to the battery -> since it doen't consider the cahrge of the battery on holidays, it maight be less than actual
            
                    daily_energy_loss_pv_forecasted [w,d] = ampl.getValue('sum{t in Time} eloss_pv[t];')
                    yearly_energy_loss_pv_forecasted  = yearly_energy_loss_pv_forecasted  + daily_energy_loss_pv_forecasted [w,d]  # amount of energy loss on pv from temporary amount of energy loss

                    daily_energy_loss_conv_forecasted [w,d] = ampl.getValue('sum{t in Time} (eapp_pv[t] + eapp_bat[t])*(1-RINV);')
                    yearly_energy_loss_conv_forecasted  = yearly_energy_loss_conv_forecasted  + daily_energy_loss_conv_forecasted [w,d] # amount of energy loss from conversion
 
                    daily_energy_loss_eapp_bat_forecasted [w,d] = ampl.getValue('SISC*RDCH/CBAT + (sum{t in 2..24} sbat[t]) * RDCH * CBAT;')
                    yearly_energy_loss_eapp_bat_forecasted  = yearly_energy_loss_eapp_bat_forecasted + daily_energy_loss_eapp_bat_forecasted [w,d]  # amount of energy loss on pv from battery self-discharging

                    daily_energy_loss_chr_forecasted [w,d] = ampl.getValue('sum{t in Time} echr[t] * (1-RCHR);')
                    yearly_energy_loss_chr_forecasted  = yearly_energy_loss_chr_forecasted  + daily_energy_loss_chr_forecasted [w,d] # amount of energy loss on pv from battery charging

                    daily_energy_loss_total_forecasted [w,d] = daily_energy_loss_pv_forecasted [w,d] + daily_energy_loss_conv_forecasted [w,d] + daily_energy_loss_eapp_bat_forecasted [w,d] + daily_energy_loss_chr_forecasted [w,d] 
                    yearly_energy_loss_total_forecasted  = yearly_energy_loss_total_forecasted  + daily_energy_loss_total_forecasted [w,d]  # calculate total amount of energy loss 
                       
                    daily_appliance_operation_hour_forecasted [w,d] = ampl.getValue ('sum{t in Time, a in Appliance} xapp_state[a,t];') # show daily pre-schedule of applainces by summing up the variable xapp_state -> in order to compare actual operation
                    yearly_appliance_operation_hour_forecasted  = yearly_appliance_operation_hour_forecasted  + daily_appliance_operation_hour_forecasted [w,d]
 


               
################ Actual solar irradiation sampling and first adjust operation of the bateery ###########################################################
 
              
                    # Define variables to describe actual solar irradiance sampling and the battery operation based on actual solar irradiance                        
                    new_SPV = {} # sampled actual solar irradiance

                    new_EPV = {} # new amount of generated solar energy based on actual solar irradiance

                    total_eapp = {} # total amount of energy used by appliances 
                   
                    avail_eapp_bat = {} # actual available amount of energy dischared from the battery          
                   
                    new_eapp_pv = {} # new amount of energy from pv to operate appliance                   
                    new_eapp_bat = {} # new amount of energy from the battery to operate appliance 

                    new_echr= {} # temporary variable to calculate actual new amount of energy chared to the battery   
                    new_echr2 = {} # actual new amount of energy chared to the battery   

                    new_eloss1_pv = {} # first temporary amount of energy loss
                    new_eloss2_pv = {} # second temporary energy loss if SOC is greater than 95% 
                   
                    pre_sbat = {} # SOC at the previous period                   
                    new_sbat = {} # temporary variable to calculate new state of charge for the battery   
                    new_sbat2 = {} # new state of charge for the battery   

                    total_new_eloss = {} # new total amount of energy loss                                  
                    new_eloss_pv = {} # new amount of energy loss on pv array : unused solar energy                               
                    new_eloss_conv = {} # new amount of energy loss during conversion
                    new_eloss_bat_dchr = {} # new amount of energy loss by battery self-discharging
                    new_eloss_bat_chr = {} # new amount of energy loss during battery charging
  
             
                    # solar irradiance sampling from period 1 to 24 to get actual solar irradiance
                    for t in time_iter :

                        new_EPV[t] = solar['s_actual'] [w,d,t] * ampl_apv * ampl_rpv # calculate amount of solar energy generated by pv array based on solar irradiance sampling                        
                        total_eapp[t] = ampl.getValue('sum{{a in Appliance}} EAPP[a] * xapp_state[a,{t}] + EM[{t}];'.format(t=t)) # calculate total amount of energy used by appliances from scheduling model
 
                        if t == 1 :
                            
                            pre_sbat[t] = ampl_sisc  # define inital SOC(%) value from AMPL to Python when t=1
                       
                        else :
                            
                            pre_sbat[t] = new_sbat2[t-1] # define SOC of the previous period t-1 when t>1
                        
                        if new_EPV[t] >= (1/ampl_rinv) * total_eapp[t] : # if actual solar energy is greater than total energy required by appliances some variables to operate appliances should be revised
                       
                            new_eapp_pv[t] = (1/ampl_rinv) * total_eapp[t] # amount of appliance required energy can only be covered by pv energy                       
                            new_eapp_bat[t] = 0 # there is no discharging                           
                            new_echr[t] = 5 - max(5 - (new_EPV[t] - new_eapp_pv[t]), 0) # amount of energy charged to the battery is 0 or > 0, but equal or less than 5kWh                           
                            new_eloss1_pv[t] = max(new_EPV[t] - new_eapp_pv[t] - new_echr[t], 0) # first temporary amount of energy loss should not be less than 0 
                            new_sbat[t] = 0.95 - max(0.95 - pre_sbat[t] * (1 - ampl_rdch) - (new_echr[t] * ampl_rchr / x_bat), 0) # SOC of the battery should be less than 95%                                                                               
                            new_eloss2_pv[t] = max(pre_sbat[t] * (1 - ampl_rdch) + (new_echr[t] * ampl_rchr / x_bat) - new_sbat[t] , 0) * (x_bat / ampl_rchr) # if SOC is greater than 95% there may be another second temporary energy loss occured                                                      
                            new_echr2[t] = new_EPV[t] - new_eapp_pv[t] - (new_eloss1_pv[t] + new_eloss2_pv[t]) # actual new amount of energy chared to the battery                            
                            new_sbat2[t] = 0.95 - max(0.95 - pre_sbat[t] * (1-ampl_rdch) - (new_echr2[t] * ampl_rchr / x_bat) , 0) # actual new SOC for the battery condiering actual amount of energy loss                           
                
                        else : # if actual solar energy is less than total energy required by appliances 
                         
                            new_eapp_pv[t] = new_EPV[t] # all actual solar energy goes to applainces
                            new_echr[t] = 0 # there is no charging
                            new_echr2[t] = 0 # there is no charging
                            new_eloss1_pv[t] = 0 # there is no energy loss
                            new_eloss2_pv[t] = 0 # there is no energy loss
                            new_sbat[t] = 0 # it is required only if sampled pv energy is greater than total energy required by appliances 
 
                            # consider the operation of the battery when there is no solar irradiance. before t = 18, the min, SOC sohuld be greater than 5%. After t=18, the min, SOC sohuld be greater than 30%                    
                            # it is important constraint. if actual solar irradiance is much less than forcasted, the model would not work without this constraints.
                            if t <= 12 :
                                
                                avail_eapp_bat[t] = min((pre_sbat[t] - ampl_smin) * x_bat, ampl_rmdc) # calculate actual available amount of energy dischared from the battery 
                                

                            else :
                                avail_eapp_bat[t] = min(max((pre_sbat[t] - ampl_send) * x_bat, 0), ampl_rmdc) # calculate actual available amount of energy dischared from the battery considering SOC of the last period of the day(more than 30%)
                            
                            avail_eapp_bat[t] = max(avail_eapp_bat[t], 0) # just in case, in order to avoid there is really really small amount of negative value occurs, max func. is used 



                      ################ Re-schedule appliance operation when there is not enough energy ###########################################################
                 
                            # Defining variables to describe rescheduling appliance operation
                            candidate_appliance_1 = {} # collect candidate of removable appliances based on priority
                            removable_appliance_1 = {} # determine removable appliance based on priority
                            candidate_appliance_2 = {} # collect candidate of removable appliances based on priority x required energy
                            removable_appliance_2 = {} # determine removable appliance based on priority x required energy                                    
                   
                            # while available amout of energy from the pv and the battery is less than energy required by appliances, appliance operation should be rescheduled until it satisfies amount of available energy.
                            while (new_eapp_pv[t] + avail_eapp_bat[t])  < (1 / ampl_rinv) * total_eapp[t] :
                                
                                candidate_appliance_1 = {} # initialize variable
                                candidate_appliance_2 = {} # initialize variable

                                # represent available amount of energy and required energy at period t                             
                                print("")    
                                print("Re-schedule of appliance operation is required at period {t}".format(t=t))
                                print("")    
                                print("After checking.....")
                                print("The amount of available energy from sampled actual solar irradiance data and the battery at period {t} is".format(t=t), (new_eapp_pv[t] + avail_eapp_bat[t]), "kWh")
                                print("The amount of energy required by appliances and energy margin at period {t} is".format(t=t), (1/ampl_rinv) * total_eapp[t], "kWh")

                                if ampl.getValue('sum{{a in Appliance}} xapp_state[a,{t}];'.format(t=t)) > 0 :
                                    
                                    # collect candidate of removable appliances which has the lowest priority
                                    for a in appliance_iter :
                                        if ampl.getValue('xapp_state[{a},{t}]'.format(a=a, t=t)) == 1 : 
                                            candidate_appliance_1[a,t] = ampl.getValue('xapp_state[{a},{t}]'.format(a=a, t=t)) * ampl.getValue('W[{a}]'.format(a=a))
                                                                         
                                    removable_appliance_1[t] = candidate_appliance_1[min(candidate_appliance_1, key=lambda x : candidate_appliance_1[x])]
       
                                    # collect candidate of removable appliances which consume less energy when the priority is equal : first remove appliance comsumes less energy when its priority is equal.                            
                                    for a in appliance_iter :
                                        if removable_appliance_1[t] == ampl.getValue('xapp_state[{a},{t}]'.format(a=a, t=t)) * ampl.getValue('W[{a}]'.format(a=a)):
                                            candidate_appliance_2[a,t] = ampl.getValue('xapp_state[{a},{t}]'.format(a=a, t=t)) * ampl.getValue('W[{a}]'.format(a=a)) *ampl.getValue('EAPP[{a}]'.format(a=a))
                                    removable_appliance_2[t] = candidate_appliance_2[min(candidate_appliance_2, key=lambda x : candidate_appliance_2[x])]
    
    
                                    # determine removable appliances and change its status as 0 at period t                           
                                    for a in appliance_iter :
                                       
                                        if removable_appliance_2[t] == ampl.getValue('xapp_state[{a},{t}]'.format(a=a, t=t)) * ampl.getValue('W[{a}]'.format(a=a)) * ampl.getValue('EAPP[{a}]'.format(a=a)):
                                           
                                            ampl.var['xapp_state'][a,t].setValue(0)         
                                           
                                            if ampl.getValue('xapp_state[1,{t}]'.format(t=t)) == 0 :
                                                
                                                ampl.var['xapp_state'][2,t].setValue(0)         
    
                                            # represent status of removed appliance
                                            print("")
                                            print("Removed appliance is {a} at period {t}".format(a=a,t=t))
                                            print("Change status of appliance {a} at period {t} : 1 to".format(a=a, t=t), ampl.getValue('xapp_state[{a},{t}]'.format(a=a, t=t)))
                                            print("Required energy of appliance {a} at period {t} is changed from".format(a=a, t=t), ampl.getValue('EAPP[{a}]'.format(a=a)), " kWh to 0")
                                           
      
                                # consider the case that available amount of energy is still less than energy margin even though there is no applaince operated during period t                                 
                                else : # ampl.getValue('sum{{a in Appliance}} xapp_state[a,{t}];'.format(t=t)) == 0 :
                                   
                                        
                                    ampl.param['EM'][t] = ((new_eapp_pv[t] + avail_eapp_bat[t])*ampl_rinv) * 0.9 # remaining energy margin -> give safety factor 10%
                                    print("")
                                    print("After adjusting, the amount of new energy margin at period {t} is".format(t=t) , ampl_rinv * ampl.getValue('EM[{t}];'.format(t=t)))


                                total_eapp[t] = ampl.getValue('sum{{a in Appliance}} EAPP[a] * xapp_state[a,{t}] + EM[{t}];'.format(t=t)) # calculating adjusted amount of energy used by re-scheduled appliances                                 
                                
                                print("") 
                                print("After adjusting, the amount of energy required by re-scheduled appliances and energy margin at period {t} is".format(t=t) , total_eapp[t])


                                # if adjusted amount of energy required by re-scheduled appliances becomes less than energy from the pv then we need to recalculate some variables
                                if new_eapp_pv[t] >= (1/ampl_rinv) * total_eapp[t] :
                                    new_eapp_pv[t] = (1/ampl_rinv) * total_eapp[t] # amount of appliance required energy can only be covered by pv energy                         
                                    new_eapp_bat[t] = 0 # there is no discharging                           
                                    new_echr[t] = 5 - max(5 - (new_EPV[t] - new_eapp_pv[t]), 0) # amount of energy charged to the battery is 0 or > 0, but equal or less than 5kwH                          
                                    new_eloss1_pv[t] = max(new_EPV[t] - new_eapp_pv[t] - new_echr[t], 0) # first temporary amount of energy loss should not be less than 0 
                                    new_sbat[t] = 0.95 - max(0.95 - pre_sbat[t] * (1 - ampl_rdch) - (new_echr[t] * ampl_rchr / x_bat), 0) # SOC of the battery should be less than 95%                                                                               
                                    new_eloss2_pv[t] = max(pre_sbat[t] * (1 - ampl_rdch) + (new_echr[t] * ampl_rchr / x_bat) - new_sbat[t] , 0) * (x_bat / ampl_rchr) # if SOC is greater than 95% there may be another second temporary energy loss occured                                                      
                                    new_echr2[t] = new_EPV[t] - new_eapp_pv[t] - (new_eloss1_pv[t] + new_eloss2_pv[t]) # actual new amount of energy chared to the battery                            
                                    new_sbat2[t] = 0.95 - max(0.95 - pre_sbat[t] * (1-ampl_rdch) - (new_echr2[t] * ampl_rchr / x_bat) , 0) # actual new SOC for the battery condiering actual amount of energy loss                           

                                print("")                   
                                print("Return to the loop to re-check")
        
                                                                              
                            new_eapp_bat[t] = (1/ampl_rinv) * total_eapp[t] - new_eapp_pv[t] # calculate adjusted amount of energy discharged from the baattery
                            new_sbat2[t] = pre_sbat[t] * (1-ampl_rdch) - (new_eapp_bat[t] / x_bat) # actual new SOC for the battery condiering adjusted amount of energy discharged from the battery                          
                 
                        new_eloss_pv[t] = new_eloss1_pv[t] + new_eloss2_pv[t]  # calculate actual amount of energy loss on pv from temporary amount of energy loss
                        new_eloss_conv[t] = (new_eapp_pv[t] + new_eapp_bat[t])*(1-ampl_rinv) # calculate actual amount of energy loss from conversion
                        new_eloss_bat_dchr[t] = pre_sbat[t]*ampl_rdch*x_bat # calculate actual amount of energy loss on pv from battery self-discharging
                        new_eloss_bat_chr[t] = new_echr2[t] * (1- ampl_rchr) # calculate actual amount of energy loss on pv from battery charging
                        total_new_eloss[t] = new_eloss_pv[t] + new_eloss_conv[t] + new_eloss_bat_dchr[t] + new_eloss_bat_chr[t]  # calculate total amount of energy loss 
                                                      
                        hourly_soc [t] = hourly_soc[t] + new_sbat2[t]


################### Dynamic profiling for washer and dryer ########################################################
                   
                    # calclate operation time of washer and dryer on day d
                    daily_opt_time_Washer [w,d] = ampl.getValue('sum{t in Time} xapp_state[1,t]')                   
                    daily_opt_time_Dryer [w,d] = ampl.getValue('sum{t in Time} xapp_state[2,t]')
                   
                    # calclate total operation time of washer and dryer on week w
                    weekly_opt_time_Washer[w] = weekly_opt_time_Washer[w] + daily_opt_time_Washer[w,d]
                    weekly_opt_time_Dryer[w] = weekly_opt_time_Dryer[w] + daily_opt_time_Dryer[w,d]

                    # if daily_opt_time_Washer[w,d] == 0 : # washer is not operated on day d
                        
                    if weekly_opt_time_Washer[w] == 0 : # washer has never been operated until day d during week w
                        
                        if d==7 :
                            priority_Washer  = param_w['w'][d_type['date_type'][w+1,1],1]
                            priority_Dryer  = param_w['w'][d_type['date_type'][w+1,1],2]


                            opt_time_Washer  = param_t['t'][d_type['date_type'][w+1,1],1] # initialize required opt.time of washer for the first day of the next week
                            opt_time_Dryer  = param_t['t'][d_type['date_type'][w+1,1],2]
    
    
                        else : # if it is not holiday and washer has never been operated until day d during week w
                             
                                  
                            priority_Washer  = param_w['w'][d_type['date_type'][w,d+1],1] + 50
                            priority_Dryer  = param_w['w'][d_type['date_type'][w,d+1],2] + 50
 
                            opt_time_Washer  = param_t['t'][d_type['date_type'][w,d+1],1]
                            opt_time_Dryer = param_t['t'][d_type['date_type'][w,d+1],2]
                               
                    
                    else : # washer is operated on day d
                     
                        if d==7 : # washer is operated once on the last day of the week
                         
                            priority_Washer  = param_w['w'][d_type['date_type'][w+1,1],1]
                            priority_Dryer  = param_w['w'][d_type['date_type'][w+1,1],2]

                            opt_time_Washer  = param_t['t'][d_type['date_type'][w+1,1],1] # initialize required opt.time of washer for the first day of the next week
                            opt_time_Dryer  = param_t['t'][d_type['date_type'][w+1,1],2]

                        else :
                            priority_Washer = 0                         
                            priority_Dryer= 0

                            opt_time_Washer = 0  # washer is no longer operated during a week
                            opt_time_Dryer = 0 # dryer is no longer operated during a week              


################### End of dynamic profiling for washer and dryer ########################################################
            
            
                    print("Available energy of the last period of day {d} of week {w} :".format(d=d, w=w), new_sbat2[24]*x_bat)  

                    soc_last_period [w,d] = new_sbat2[24]
                    ampl_sisc = soc_last_period [w,d]

                    daily_energy_generated_actual [w,d] = sum(new_EPV.values())  
                    yearly_energy_generated_actual  = yearly_energy_generated_actual  + daily_energy_generated_actual [w,d]
                   
                    #yearly energy demand
                    daily_energy_demand_actual [w,d] = ampl.getValue ('sum{a in 3..25} EAPP[a] * T[a] + sum{t in Time} EM[t];')
                    yearly_energy_demand_actual  = yearly_energy_demand_actual  + daily_energy_demand_actual [w,d]
                              
                    #yearly energy used by appliances
                    daily_energy_appliance_total_actual [w,d] = sum(total_eapp.values())       
                    yearly_energy_appliance_total_actual   = yearly_energy_appliance_total_actual   + daily_energy_appliance_total_actual[w,d] 
                   
                    #yearly energy used by appliance from pv    
                    daily_energy_appliance_pv_actual [w,d] = sum(new_eapp_pv.values()) * ampl_rinv                                  
                    yearly_energy_appliance_pv_actual   = yearly_energy_appliance_pv_actual   + daily_energy_appliance_pv_actual [w,d] 
                   
                    #yearly energy used by appliance from battery
                    daily_energy_appliance_battery_actual [w,d] = sum(new_eapp_bat.values()) * ampl_rinv       
                    yearly_energy_appliance_battery_actual   = yearly_energy_appliance_battery_actual   + daily_energy_appliance_battery_actual [w,d]
                    
                    #yearly energy charged to the battery
                    daily_energy_charged_actual [w,d] = sum(new_echr2.values())      
                    yearly_energy_charged_actual  = yearly_energy_charged_actual   + daily_energy_charged_actual [w,d]
                   
                    #yearly energy loss on pv
                    daily_energy_loss_pv_actual [w,d] = sum(new_eloss_pv.values())    
                    yearly_energy_loss_pv_actual  = yearly_energy_loss_pv_actual   + daily_energy_loss_pv_actual [w,d] 
                                   
                    #yearly energy loss from conversion
                    daily_energy_loss_conv_actual [w,d] = sum(new_eloss_conv.values())
                    yearly_energy_loss_conv_actual   = yearly_energy_loss_conv_actual  + daily_energy_loss_conv_actual [w,d]
                    
                    #yearly energy loss from the battery self discharging
                    daily_energy_loss_eapp_bat_actual [w,d] = sum(new_eloss_bat_dchr.values())       
                    yearly_energy_loss_eapp_bat_actual   = yearly_energy_loss_eapp_bat_actual   + daily_energy_loss_eapp_bat_actual [w,d]

                    #yearly energy loss from battery charging
                    daily_energy_loss_chr_actual [w,d] = sum(new_eloss_bat_chr.values()) 
                    yearly_energy_loss_chr_actual   = yearly_energy_loss_chr_actual   + daily_energy_loss_chr_actual [w,d]      

                    #yearly energy loss
                    daily_energy_loss_total_actual [w,d] = sum(total_new_eloss.values())       
                    yearly_energy_loss_total_actual   = yearly_energy_loss_total_actual   + daily_energy_loss_total_actual [w,d]
                                     

                    daily_appliance_operation_hour_actual [w,d] = ampl.getValue ('sum{t in Time, a in Appliance} xapp_state[a,t];') # show daily pre-schedule of applainces by summing up the variable xapp_state -> in order to compare actual operation
                    yearly_appliance_operation_hour_actual   = yearly_appliance_operation_hour_actual   + daily_appliance_operation_hour_actual [w,d]

                   # calclate operation time of washer and dryer on day d
                    yearly_opt_time_Washer   = yearly_opt_time_Washer   + daily_opt_time_Washer [w,d]
                   
                    yearly_opt_time_Dryer   = yearly_opt_time_Dryer  + daily_opt_time_Dryer [w,d]


                    # Calculate total operation time of each applaince based on the type of the day for the anlayis purpose            
                    for a in appliance_iter :
                        for t in time_iter :
                            if (ampl.param['DATE'][w,d] == 'workday_winter') or (ampl.param['DATE'][w,d] == 'workday_summer') : 
                                total_op_time_applaince_weekday[a] = total_op_time_applaince_weekday[a] + ampl.getValue('xapp_state[{a},{t}]'.format(a=a,t=t))
                            if (ampl.param['DATE'][w,d] == 'weekend_winter') or (ampl.param['DATE'][w,d] == 'weekend_summer') :                        
                                total_op_time_applaince_weekend[a] = total_op_time_applaince_weekend[a] + ampl.getValue('xapp_state[{a},{t}]'.format(a=a,t=t))
                            if ampl.param['DATE'][w,d] == 'holiday' :                        
                                total_op_time_applaince_holiday[a] = total_op_time_applaince_holiday[a] + ampl.getValue('xapp_state[{a},{t}]'.format(a=a,t=t))
             
           
            yearly_energy_demand_total_actual = yearly_energy_demand_actual + ampl.getValue ('EAPP[1] * 2 + EAPP[2] * 1;') * (w-1) # actual energy demand = appliance 3-25 *364 days + applaince 1-2 * 51weeks(except for the holiday week)
            
            
            # Calculate Avg. hourly SOC for the anlayis purpose
            for t in time_iter :
                avg_hourly_soc[t] = hourly_soc[t] / ((w-1)*7 + d)
                
                

            # After running the energy scheduling model and simulation model, d_loss will be calculated and send it to the algorithm.            

            if decide_running == 0 :
            
                annual_demand_satisfaction  = yearly_energy_appliance_total_actual / yearly_energy_demand_total_actual 
                annual_demand_loss_actual = yearly_energy_demand_total_actual - yearly_energy_appliance_total_actual 
                d_loss_sim =  max(annual_demand_loss_actual,0)  
              
            # Print the model running results.

            print(" ") 
            print("Running successful!")    
            print("PV size : ",  x_pv, "and Battery capacity :", x_bat, " satisfy model running condition.")    
            print("Amount of non-served energy ", d_loss_sim ,"for the solution (", x_pv, ",", x_bat, "): ")  

            # initialize initial SOC for the next pv-Battery combination
            ampl.param['SISC'] = 0.30
            
            print("Total computational time: ", time.perf_counter() - start)  

