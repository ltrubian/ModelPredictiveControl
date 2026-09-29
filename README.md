# ModelPredictiveControl

## Brief description
This repository is the working folder of the final project of University course **Adaptive and Model Predictive Control**.

The main goal of the project is to familiarize with the *ACADOS* library, directly observe potential and weakness of the MPC approach, and noticing the caveat of simulations in the validation step of a controller design.

More details in *docs/Description.pdf*

The results are in *ModelPredictiveControl.pdf*

## Examples of usage

### Help

The `help` is a good point to see what kind of comparison are already available in *scenarios.py*.

     python3 main.py --help
     # uv run main.py --help
     
output

    usage: main.py [-h] [-s {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}] [-c CASE] [-a] [--save-fig]
    
    It runs simulaitn of inverted pendulum in some predefined and tested scenarios
    
    options:
      -h, --help            show this help message and exit
      -s {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, --simulation {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
                            select simulation to run
      -c CASE, --case CASE  select one of the available set of parameters
      -a, --analysis        make comparison of all the cases for the simulation 's'('c' will be ignored)
      --save-fig            save figures in ./images/sim_<simulation>/
    
    Here the possible choices of arguments
     -s  0 -c [0-1] -> task 1, controllers comparison
                     varying: ['mod_type_ocp', 'Q']
     -s  1 -c [0-2] -> task 1, default non-linear
                     varying: ['R']
     -s  2 -c [0-2] -> task 1, extended controlloer
                     varying: ['Q']
     -s  3 -c [0-2] -> task 1, extended controlloer
                     varying: ['R']
     -s  4 -c [0-3] -> task 2, simulate default system (ERK)
                     varying: ['ts_sim']
     -s  5 -c [0-3] -> task 2, simulate default system (IRK)
                     varying: ['ts_sim']
     -s  6 -c [0-1] -> task 2, simulate default system (ERK vs IRK)
                     varying: ['integ_type_sim']
     -s  7 -c [0-1] -> task 2, simulate default system (ERK vs IRK)
                     varying: ['integ_type_sim']
     -s  8 -c [0-3] -> task 2, simulate spring system (ERK)
                     varying: ['ts_sim']
     -s  9 -c [0-3] -> task 2, simulate spring system (IRK)
                     varying: ['ts_sim']
     -s 10 -c [0-1] -> task 2, simulate spring system (ERK vs IRK)
                     varying: ['integ_type_sim']
     -s 11 -c [0-1] -> task 2, simulate spring system (ERK vs IRK)
                     varying: ['integ_type_sim']
     -s 12 -c [0-1] -> task 2, default non-linear controller, default sampling time
                     varying: ['integ_type_ocp']
     -s 13 -c [0-1] -> task 2, default non-linear controller, higher sampling time
                     varying: ['integ_type_ocp']
     -s 14 -c [0-1] -> task 3, NMPC vs linear MPC
                     varying: ['mod_type_ocp']
     -s 15 -c [0-1] -> task 3, swing-up maneuver with linear MPC
                   varying: ['mod_type_ocp']

### Single run

A single run allows to investigate a specific configuration of paramters

    python3 main.py -s 2 -c 1

with output containing both summary of parameters and results (plus plots)

    Simulation task 1, extended controlloer with parameters: 
    {'ini_type': 'down',
     'ref_type': 'swing-up',
     'mod_type_ocp': 'extended',
     'R': 0.01,
     'Q': array([[10. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. , 10. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0.1,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0.1,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0.1]])}
    Simulation: 100%|██████████████████| 15000/15000 [00:04<00:00, 3024.97it/s]
    Average total controller CPU time: 2.1523586666666668 ms
    Average total simulation CPU time: 0.001667266666666667 ms
    ##### state 0 ######
             labels underpeak Upeak time       peak  peak time  overshoot  rise time settl time 
                0.1      0.99       6.58       0.99       6.58        inf       0.00      15.00 
    ##### state 1 ######
             labels underpeak Upeak time       peak  peak time  overshoot  rise time settl time 
                0.1     31.60       5.56       3.83       7.70       2.13       0.94       7.85 
    ##### state 2 ######
             labels underpeak Upeak time       peak  peak time  overshoot  rise time settl time 
                0.1      3.34       6.04       3.34       6.04        inf       0.00      15.00 
    ##### state 3 ######
             labels underpeak Upeak time       peak  peak time  overshoot  rise time settl time 
                0.1   -321.65       5.94     321.65       5.94        inf       0.00      15.00 

### Multiple runs

While the flag `-a` (or `--analysis`) allows a **comparison** varying a specific parameter

    python3 main.py -s 2 -a  

with output containing both summary of parameters and results (plus plots)

    Simulation task 1, extended controlloer with parameters: 
    {'ini_type': 'down',
     'ref_type': 'swing-up',
     'mod_type_ocp': 'extended',
     'R': 0.01,
     'Q': array([[10.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  , 10.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.1 ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.1 ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.01]])}
    Simulation: 100%|██████████████████| 15000/15000 [00:04<00:00, 3019.11it/s]
    Average total controller CPU time: 2.1059333333333337 ms
    Average total simulation CPU time: 0.0016797333333333337 ms
    Simulation task 1, extended controlloer with parameters: 
    {'ini_type': 'down',
     'ref_type': 'swing-up',
     'mod_type_ocp': 'extended',
     'R': 0.01,
     'Q': array([[10. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. , 10. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0.1,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0.1,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0.1]])}
    Simulation: 100%|██████████████████| 15000/15000 [00:04<00:00, 3031.17it/s]
    Average total controller CPU time: 2.1570253333333334 ms
    Average total simulation CPU time: 0.0016970666666666669 ms
    Simulation task 1, extended controlloer with parameters: 
    {'ini_type': 'down',
     'ref_type': 'swing-up',
     'mod_type_ocp': 'extended',
     'R': 0.01,
     'Q': array([[10. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. , 10. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0.1,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0.1,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0.2]])}
    Simulation: 100%|██████████████████| 15000/15000 [00:05<00:00, 2933.89it/s]
    Average total controller CPU time: 2.253010666666667 ms
    Average total simulation CPU time: 0.0017136000000000002 ms
    
    Solver returned non-zero status at the following iterations:
    * k = 250 [t = 5.000 s] - status 2
    ##### state 0 ######
             labels underpeak Upeak time       peak  peak time  overshoot  rise time settl time 
               0.01      1.14       6.50       1.14       6.50        inf       0.00      15.00 
                0.1      0.99       6.58       0.99       6.58        inf       0.00      15.00 
                0.2      0.86       6.67       0.86       6.67        inf       0.00      15.00 
    ##### state 1 ######
             labels underpeak Upeak time       peak  peak time  overshoot  rise time settl time 
               0.01     28.78       5.54       5.48       7.52       3.05       0.93       7.89 
                0.1     31.60       5.56       3.83       7.70       2.13       0.94       7.85 
                0.2     33.71       5.58       2.92       7.88       1.62       0.94       7.22 
    ##### state 2 ######
             labels underpeak Upeak time       peak  peak time  overshoot  rise time settl time 
               0.01      3.54       6.00       3.54       6.00        inf       0.00      15.00 
                0.1      3.34       6.04       3.34       6.04        inf       0.00      15.00 
                0.2      3.17       6.08       3.17       6.08        inf       0.00      15.00 
    ##### state 3 ######
             labels underpeak Upeak time       peak  peak time  overshoot  rise time settl time 
               0.01   -317.91       5.90     317.91       5.90        inf       0.00      15.00 
                0.1   -321.65       5.94     321.65       5.94        inf       0.00      15.00 
                0.2   -324.15       5.97     324.15       5.97        inf       0.00      15.00 
    
    
