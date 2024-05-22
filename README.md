# normal-mode-exporter

**Note**: This script now lives in the https://github.com/KoehnLab/python-scripts repository.

Python utility for exporting normal modes from quantum chemistry application output to more easily interpretable formats.

Currently supported input formats:
- [TurboMole](https://www.turbomole.org/) calculation directory (the one containing the `control` file)
- [Molpro](https://www.molpro.net/) output file

From the obtained data, the program will export
- The equilibrium geometry as a XYZ file
- The displacement vector for every normal mode (as `.displacement` file)
- An animation of every normal mode as a XYZ file containing multiple frames (use e.g. [vmd](https://www.ks.uiuc.edu/Research/vmd/) for visualization)

## Usage

```
usage: normal_mode_exporter.py [-h] --input PATH --output PATH [--input-format {auto,turbomole}] [--step-size SIZE] [--displacement-scaling FACTOR]

Export vibrational normal modes to easily accessible formats

optional arguments:
  -h, --help            show this help message and exit
  --input PATH, -i PATH
                        Path to the input file or directory
  --output PATH, -o PATH
                        Path to the output directory
  --input-format {auto,turbomole}
                        Specify the format of the provided input
  --step-size SIZE      The step size to use for animations
  --displacement-scaling FACTOR
                        Scaling factor to apply to displacement vectors during animation
```

