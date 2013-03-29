Kernel-Based-Object-Tracking
============================

This project is the C++ implementation of kernel based object tracking as discussed by Visvanathan Ramesh, Dorin Comaniciu &amp; Peter Meer in their paper "Kernel-Based Object Tracking".

In this project the objects are represented by their color histograms weighted by isotropic kernel. Targets (or objects) are compared in subsequent frames to calculate the bhattacharya distance which is subsequently used to move the tracked using mean shift.

Note: This project was implemented in visual studio, thus remove conio and getch to be able to use in other environments.
