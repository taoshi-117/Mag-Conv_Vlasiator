# Mag-Conv_Vlasiator
The code for quantifying magnetospheric convection in Vlasiator simulation.


The Magnetospheric Convection is a crucial topic in space physics.
In this code we are using the results of Vlasiator to quntify the Magnetospheric Convection (i.e., Dungey cycle) in the simulation:

1.  Calculate the closed flux in the simulation.
2.  Retrieve the azimuthal and radial convection rate in each MLT sector in the closed field line region.
3.  Subtract 1 and 2 to get the magnetic reconnection rate in each sector.
4.  The Dungey cycle convection rate (Dccr) can be represented by subtracting dayside and nightside reconnection rate.
5.  Confirm the Dccr by the open flux change rate in the polar cap.
