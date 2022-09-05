This folder contains code used to produce the figures in Chapter 5 of the thesis, *Bayesian Probabilistic Numerical Methods for
Ordinary and Partial Differential Equations*. Code for this chapter is coded in Python, except for one Matlab script. Python libraries required are numpy, matplotlib and pandas.

all Python scripts contained in this folder can be run on the command line (Windows or Linux), for example:

```
python porousmedium.py
```

To reproduce the error and z score plots for the Homogeneous Burger's Equation (Figure 5.2), go to the Burgernoforce folder. Run the Python script Burgernoforce.py, which will produce an output text file containing the errors and z scores, once the script has finished running, subsequently run the Python script Burgernoforcegraph.py, and the required Figures will be saved as pdf files.

Similarly to reproduce the error and z score plots for the Homogeneous Burger's Equation for the rational quadratic kernel (Figure 5.3), go to the Burgernoforcerational folder. Run the Python script Burgernoforcerational.py, which will produce an output text file containing the errors and z scores, once the script has finished running, subsequently run the Python script Burgernoforcegraphrational.py, and the required Figures will be saved as pdf files.

Similarly to reproduce the error and z score plots for the Porous medium equation (Figure 5.4), go to the Porousmedium folder. Run the Python script porousmedium.py, which will produce an output text file containing the errors and z scores, once the script has finished running, subsequently run the Python script Porousgraph.py, and the required Figures will be saved as pdf files.

Similarly to reproduce the error and z score plots for the Porous medium equation for the alternate linearisation (Figure 5.5), go to the Porousmediumalt folder. Run the Python script porousmediumalt.py, which will produce an output text file containing the errors and z scores, once the script has finished running, subsequently run the Python script Porousaltgraph.py, and the required Figures will be saved as pdf files.

Similarly to reproduce the error and z score plots for the Porous medium equation with conservation of mass (Figure 5.6), go to the Porousmediumintegral folder. Run the Python script porousmediumintegral.py, which will produce an output text file containing the errors and z scores, once the script has finished running, subsequently run the Python script Porousintegralgraph.py, and the required Figures will be saved as pdf files.

To reproduce the error and z score plots for the Forced Burger's Equation (Figure 5.7) is a little more involved. Go to the Burgerforce folder. First, the matlab script 'burgerpde.m' contains a function that needs to be run with the input 'burgerpde(513,513)' in the Matlab command window. This will generate a solution from Matlab's in built PDE solver pdepde for the Forced Burger's Equation, which will be written to a text file 'burgerforcedmatlab1.txt'. Once this is done, the Python files can be run in the same way as all the others. So run the Python script Burgerforced.py, which will produce an output text file containing the errors and z scores, once the script has finished running, subsequently run the Python script Burgerforcedgraph.py, and the required Figures will be saved as pdf files. Similarly, to generate error plots from the Crank Nicolson scheme, run the Python script CrankNicholsonBurger.py first, and then CrankNicholsongraph.py after.

To reproduce the surface plot (Figure 5.1), go to the Burgernoforce folder. Open the Python script Burgernoforce.py, change 

```
Nset=[5,9,17,33,65,129]
Mset=[5,9,17,33,65,129]
```

to

```
Nset=[65]
Mset=[65]
```

Run the script, after the script is finished, open surfaceplot.py (inside an IDE like Pycharm), run it and the surface plot in Figure 5.1 will be generated. Note this surface plot will not be saved automatically, but you can manually save it from the pop up. 
