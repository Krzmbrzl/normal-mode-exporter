# normal-mode-exporter

Python utility for exporting normal modes from quantum chemistry application output to more easily interpretable formats.

Currently supported input formats:
- TurboMole calculation directory

From the obtained data, the program will export
- The equilibrium geometry as a XYZ file
- The displacement vector for every normal mode (as `.displacement` file)
- An animation of every normal mode as a XYZ file containing multiple frames (use e.g. [vmd](https://www.ks.uiuc.edu/Research/vmd/) for visualization)

