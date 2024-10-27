This directory provides the modified FMUs used to gather the data for the training process of the StateChecker neural network.

Controller.FMU is the baseline Controller, while Controller_new.FMU has been extended providing the possibility to model failure in the temperature measurements. It has a couple more parameters:

* Fault: An integer that if set to 0 means that there is no fault; 1 is a constant increase or decrease of the measured value; 2 is a increase of decrease of the read value by a value obtained from a gaussian normal; 3 its a constantly increasing value with a slope that can be defined at cosimulation time.
* Fault_value: indicates the constant value, the standard deviation or the slope of the fault depending on the value of "Fault"
* Fault_start_time: the time in which the fault begins

Finally, those FMUs have been used to gather data with the DSE tool, in order to train and develop a neural network able to detect the presence or absence of faults in the system.
Said neural network has been exported into an FMU, the StateChecker.FMU, using pytorch.
