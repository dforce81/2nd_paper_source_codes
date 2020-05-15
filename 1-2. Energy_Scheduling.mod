## Week = index for week 1 to day 52
set Week;

## Day = index for day 1 to day 365
set Day;

## Scenario = index for type of the day
set Scenario;

## Time = index for time period, at the end of ...
set Time ;

## Appliance = index for appliances
set Appliance;

#########################################################################################


### Set parameters
param A {t in Time} >= 0;

param Eloss_pv {t in Time} >= 0 ;

param M ;

## DATE = Type of date 
param DATE {w in Week, d in Day} symbolic;

## APV = Area of PV array (m^2) => it means the capacity of 5.6kw
param APV ;

## RPV = PV array efficiency rate 
param RPV ;  

## RINV = Inverter efficiency rate
param RINV  ;

## EM = Energy margin to operate occasionally used appliances (kWh)
param EM {t in Time} >=0 ;
param EM_SCENARIO {s in Scenario, t in Time} >=0 ;

## SPV = Forecasted solar radiation incident on the PV array during period t of day d (kWh/m^2)
param SPV {t in Time} >=0 ;
param SPV_FORECASTED {w in Week, d in Day, t in Time} >=0 ;
param SPV_ACTUAL {w in Week, d in Day, t in Time} >=0 ;

## RDCH = Battery self-discharging efficiency rate
param RDCH  ;

## RCHR = Battery charging efficiency rate
param RCHR  ;

## CBAT = Battery capacity (kWh)
param CBAT ;

## SMIN = Battery minimum state of charge (%)
param SMIN  ;

## SMAX = Battery maximum state of charge (%)
param SMAX ;

## SISC : Battery initial state of charge (%)
param SISC ;

## SISC : Battery state of charge at the last period of the day(%)
param SEND ;

## RMCH = Battery maximum charging energy (kWh)
param RMCH ;

## RMDC =  Battery maximum discharging energy (kWh)
param RMDC ;

## EAPP = Energy consumption of appliance a (kWh)
param EAPP {a in Appliance} >= 0 ;
param EAPP_SCENARIO {s in Scenario, a in Appliance} >=0 ;

## T = Required number of periods to operate appliance a
param T {a in Appliance} >= 0 ;
param T_SCENARIO {s in Scenario, a in Appliance} >=0 ;

## W = Priority weight (1, 2, ...,10) of appliance a 
param W {a in Appliance} >= 0 ;
param W_SCENARIO {s in Scenario, a in Appliance} >= 0 ;

## P = Time preference (0/1) to operate appliance a during period t
param P {a in Appliance,t in Time} >= 0 ;
param P_SCENARIO {s in Scenario, a in Appliance, t in Time} >=0 ;


#########################################################################################


### Set variables
var new_echr{t in Time} >= 0 ;

var new_eloss1{t in Time} >= 0 ;

## EPV = Forecasted PV array energy during period t (kWh)
var EPV {t in Time} >= 0 ;

## eapp_bat = Energy discharged from the battery to operate the appliances during period t (kWh)
var eapp_bat {t in Time} >= 0 ;

## echr = Energy from the PV array used to charge the battery during period t (kWh)
var echr {t in Time} >= 0  ;

## eloss_pv = energy remaining after charging the battery and meeting the demand during time t on day i
var eloss_pv {t in Time} >= 0 ;
var total_new_eloss {t in Time} >= 0 ;

## eapp_pv = Energy from the PV array to operate the appliances during period t (kWh)
var eapp_pv {t in Time} >= 0 ;

## sbat = Battery state of charge during period t (%)
var sbat {t in Time} >= 0 ;
var new_sbat {t in Time} >= 0 ;

## ychr = Binary variable to indicate that the battery is charging during period t : binary (0,1)
var ychr {t in Time} binary ;

## ydch = Binary variable to indicate if the battery is discharging during period t : binary (0,1)
var ydch {t in Time} binary ;

## xapp_state = Binary variable to indicate the operating state of appliance a during period t : binary (0,1)
var xapp_state {a in Appliance, t in Time} binary ;

## xapp_state = Binary variable to indicate the inoperative state of appliance a during period t : binary (0,1)
var xapp_inop {a in Appliance, t in Time} binary ;

## xapp_end = Binary variable to indicate that appliance a finished operation at the end of period t : binary (0,1)
var xapp_end {a in Appliance, t in Time} binary ;

## z = Binary variable to determine if appliance a is operated or not : binary (0,1)
var z{a in Appliance} binary;



#########################################################################################


### Set Objective function

## Maximize the use of appliances which has higher priorities :

maximize appliances :


sum{t in Time, a in Appliance} W[a]* xapp_state[a,t] ;



#########################################################################################


## Set Constraints

## PV Energy Output 1
subject to PVEnergyOutput1 {t in Time}  :
EPV[t] =  RPV * SPV[t] * APV ;

## PV Energy Output 2
subject to PVEnergyOutput2 {t in Time}  :
eapp_pv[t] + echr[t] + eloss_pv[t]  = EPV[t]  ;
 
## Total Energy Consumed
subject to TotalEnergyConsumed1 {t in Time} :
sum{a in Appliance} (EAPP[a] * xapp_state[a,t]) + EM[t] = (eapp_pv[t] + eapp_bat[t])*RINV;
#sum{a in Appliance} (EAPP[a] * xapp_state[a,t])  = (eapp_pv[t] + eapp_bat[t])*RINV;

## Battery Charge and Discharge 1
subject to BatteryChargeandDischarge1 {t in Time} :
sbat[1] = SISC*(1-RDCH)  + (echr[1]*RCHR)/CBAT - eapp_bat[1]/CBAT;

subject to BatteryChargeandDischarge2 {t in 2..24} :
sbat[t] = sbat[t-1]*(1-RDCH) + (echr[t]*RCHR)/CBAT - eapp_bat[t]/CBAT ;

## Battery Charge and Discharge 3
subject to BatteryConstraint3 {t in Time} :
SMIN <= sbat[t] ; # original model constraint

## Battery Charge and Discharge 4
subject to BatteryConstraint4 {t in Time} :
sbat[t] <= SMAX ;

## Battery Charge and Discharge 5
subject to BatteryConstraint5 {t in Time} :
ychr[t] + ydch[t] <= 1 ;

## Battery Charge and Discharge 6
subject to BatteryConstraint6 {t in Time} :
eapp_bat[t] <= ydch[t]*RMDC ;

## Battery Charge and Discharge 7
subject to BatteryConstraint7 {t in Time} :
echr[t] <= ychr[t]*RMCH ; 

## Battery Charge and Discharge 7
subject to BatteryConstraint8 {t in Time} :
sbat[24] >= SEND  ; 


### Scheduling Constraints ###


## Appliance Daily Operation 1
subject to ApplianceDailyOperation1 {a in Appliance}:
sum{t in Time} xapp_state[a,t] <= T[a] ;

## Operation Period of Uninterruptible Appliance 
subject to OperationPeriodofUninterruptibleAppliance {a in 1..5}:
sum{t in Time} xapp_state[a,t] = T[a]*z[a] ;

## Operation Period Preference 
subject to OperationPeriodPreference {a in Appliance, t in Time}:
xapp_state[a,t] <= P[a,t] ;

## Uninterruptible Operation Constraints 3-1
subject to UninterruptibleOp1 {a in 1..5, t in Time}:
xapp_state[a,t] <= 1-xapp_end[a,t] ;

## Uninterruptible Operation Constraints 3-2
subject to UninterruptibleOp2 {a in 1..5, t in 1..23}:
xapp_state[a,t] - xapp_state[a,t+1] <= xapp_end[a,t+1] ;

## Uninterruptible Operation Constraints 3-3
subject to UninterruptibleOp3 {a in 1..5, t in 1..23}:
xapp_end[a,t] <= xapp_end[a,t+1] ;


## Sequential Processing of Interruptible Appliances Constraints, when appliance 1 starts earlier than applaince 2

subject to xapp_endProcess :
xapp_state[2,1]=0;

subject to xapp_endProcess00 {a in 1..2, t in 1..23}:
T[1] - sum{n in 1..t} xapp_state[1,n] <=  M*(1-xapp_state[2,t+1]) ;




