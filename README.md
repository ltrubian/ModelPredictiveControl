# ModelPredictiveControl

## Brief description
This repository is the working folder of the final project of University course **Adaptive and Model Predictive Control**.

The main goal of the project is to familiarize with the *ACADOS* library, directly observe potential and weakness of the MPC approach, and noticing the caveat of simulations in the validation step of a controller design.

More details in *docs/Description.pdf*

 ---

## TODO task 1

Consider separation. Take care of:

  - [] plot of correct input of the system, states, and *derivative* of the input 
  - [] storing variable for the real input of the system and the *derivative*: this as to take into consideration
       the fact that **simulation** states/input and **controller** states/input are different
  - [] cost function has a strong impact, consider different weights for *F*
