# Barcodes-verification
Verification of linear barcodes print quality according to ISO/IEC15416 specifications, using Image Processing and Computer Vision techniques.

For more theoretical information, check out the following documents.
- `Linear Barcodes Verification Project.pdf`: description of the project assignment.
- `guide-barcode-verification.pdf`: description of linear barcodes print quality verification.
- `report.pdf`: description of the solution.

## Description
Given an image containing a barcode, the task consists in computing some print quality parameters about the barcode. 

Since the the barcode in the input image can be rotated and can have different scales and since the input image can contain other objects apart from the barcode, the quality parameters must be computed on a standardized image. We refer to this image as "refined ROI image", since it perfectly fits the Region Of Interest (i.e. the barcode) and it is refined according to some standards. More specifically, the refined ROI image is the sub-image of the input image which has the following properties.
- It contains the barcode, and the bars are perfectly vertical.
- Along the width, there are exactly $10*X$ pixels before the first barcode bar and after the last barcode bar, where $X$ is the minimum width of a bar.
- Along the height, it perfectly fits the bar with smallest height. Basically, the height of the refined ROI image is equal to the minimum height of a barcode bar. 

<p align="left">
  <img width="150vw" src="./images/original_image_22.png">
  <img width="150vw" src="./images/refined_roi_image_quantities_22.png">
  <img width="150vw" src="./images/refined_roi_image_22.png">
</p>

Then, the print quality parameters are computed. For computing the quality parameters, $10$ equally spaced horizontal lines are considered in the refined ROI image.
<p align="center">
  <img width="150vw" src="./images/refined_roi_image_scanlines_22.png">
</p>

The quality parameters are computed one each scanline, by considering the *scan reflectance profile*, and they are the following.
- Minimum reflectance, i.e. $R_{\text{min}}$.
- Symbol Contrast, i.e. $SC$. For computing it, also the maximum reflectance, i.e. $R_{\text{max}}$, is taken into account.
- Minimum Edge Contrast, i.e. $EC_{\text{min}}$.
- MODULATION.
- DEFECT. For computing it, also the maximum Element Reflectance Non-uniformity, i.e. $ERN_{\text{max}}$, is taken into account.
<p align="center">
  <img width="300vw" src="./images/scanlines_scanReflectanceProfiles_22.png">
</p>

For each of these parameters, a numerical value is computed, and a symbolic grade between 'A' and 'F' is assigned, by using specific rules ('A' means very good, 'F' means very bad). 
In addition, a symbolic grade and a numerical value are assigned to the whole scanline.

Finally, an overall symbolic grade and an overall numerical value are assigned to the whole barcode.

### Dataset
By default, the `execute_*.py` scripts create a graphical representation of the solution. This is achieved thanks to the auxiliary `visualize.py` script.

By default, the scripts of the *compare models* group create a comparison plot. of the solution. This is achieved thanks to the auxiliary `plot_comparisons.py` script.

# Models list
The names of the models and encodings presented inside the `src` subfolders are numerical and therefore not easy to understand. The files `MODELS RECAP.md` and `ENCODINGS RECAP.md` inside the `src` subfolders provide a description of each model (or encoding) name.

The following is the list of the best models (or encodings) for each approach:
* **CP**:
  - Without rotation: `model_6D1`
  - With rotation: `model_r_7B`
* **LP**:
  - Without rotation: `model_grid`, *CPLEX* solver and *symmetry breaking* applied
  - With rotation: `model_r_0`, *Gurobi* solver and *symmetry breaking* applied
* **SAT**:
  - Without rotation: `encoding_10B`
  - With rotation: `encoding_11B`
* **SMT**:
  - Without rotation: `encoding_2C`, *z3* solver
  - With rotation: `encoding_5B`, *z3* solver

## Usage
### Execute models
```sh
# Execute instance "ins-3" with CP "model_6D1"
python src/scripts/execute_cp.py model_6D1 ins-3 --time-limit 300
```
```sh
# Execute instance "ins-3" with LP "model_1" and solver "Gurobi"
python src/scripts/execute_lp.py model_1 ins-3 gurobi --time-limit 300
```
```sh
# Execute instance "ins-3" with SAT "encoding_10B"
python src/scripts/execute_sat.py encoding_10B ins-3 --time-limit 300
```
```sh
# Execute instance "ins-3" with SMT "encoding_2C" and solver "z3"
python src/scripts/execute_smt.py encoding_2C ins-3 z3 --time-limit 300
```

### Compare models
```sh
# Compare the results of the first 10 instances with CP models "model_6D1" and "model_r_7B"
python src/scripts/compare_cp_models.py --models-list model_6D1 model_r_7B -lb 1 -ub 10
```
```sh
# Compare the results of the first 10 instances with LP models "model_1" and "model_r_0" and solvers "CPLEX" and "Gurobi"
python src/scripts/compare_lp_models.py --models-list model_1 model_r_0 --solvers-list cplex gurobi -lb 1 -ub 10
```
```sh
# Compare the results of the first 10 instances with SAT encodings "encoding_10B" and "encoding_11B"
python src/scripts/compare_sat_encodings.py --encodings-list encoding_10B encoding_11B -lb 1 -ub 10
```
```sh
# Compare the results of the first 10 instances with SMT encodings "encoding_2C" and "encoding_5B" and solvers "z3" and "cvc5"
python src/scripts/compare_smt_encodings.py --encodings-list encoding_2C encoding_5B --solvers-list z3 cvc5 -lb 1 -ub 10
```
### Solve all instances
```sh
# Solve all instances accounting for the rotation of the circuits with the best model for solver CP. 
python src/scripts/solve_all_instances.py cp --rotation
```

## Dependencies
It is required for the execution of the CP models to install [_MiniZinc_](https://www.minizinc.org/doc-2.2.3/en/installation.html) and add the executable to the environment variable PATH. 

To execute SAT the *Z3* theorem prover for python is required. 
The simplest way to install it is to use Python's package manager pip:
```sh
pip install z3-solver
```

The SMT solvers executables are already present in the directory `src/smt/solvers`.

For LP the [_AMPL_](https://ampl.com/products/ampl/) software and license are required. Moreover at least one of the following solvers is needed: [_Gurobi_](https://www.gurobi.com/products/gurobi-optimizer/), [_CPLEX_](https://www.ibm.com/analytics/cplex-optimizer) and [_Cbc_](https://github.com/coin-or/Cbc). Note that some scripts require the installation of *Gurobi* or *CPLEX*. Finally, the installation of the *amplpy* library is necessary. It can easily be installed through pip:
```sh
pip install amplpy
```

If not already installed Python libraries *pandas* and *Numpy* shall be installed.

## Interfaces

### Execute models
The `execute_*.py` scripts all present the following positional arguments:
* `model`: The model to execute (`encoding` for *SAT* and *SMt*)
* `instance`: The instance to solve

And the following optional parameters:
* `output-folder-path`: The path in which the output file is stored
* `output-name`: The name of the output solution
* `--time-limit`: The allowed time to solve the task in seconds
* `--no-create-output`: Skip the creation of the output solution
* `--no-visualize-output`: Skip the visualization of the output solution (defaults as true if `--no-create-output` is passed).

Moreover `execute_lp.py` presents the following parameters:
* `solver`: The solver used for optimization
* `--use-symmetry`: Break symmetries in the presolve process.
* `--use-dual`: Use the dual model.
* `--use-no-presolve`: Avoid AMPL presolving process.

Finally, `execute_smt.py` presents the following parameter:
* `solver`: The solver used for optimization.

### Compare models
The `compare_*.py` scripts all present the following positional arguments:
* `output-name`: The name of the output solution
* `output-folder-path`: The path in which the output file is stored

And the following optional parameters:
* `--models-list`: List of models to compare
* `--instances-lower-bound`: Lower bound of instances to solve (default 1)
* `--instances-upper-bound`: Upper bound of instances to solve (default 40)
* `--no-visualize`: Do not visualize the obtained comparisons

Moreover `compare_lp_models.py` presents the following parameters:
* `--solvers-list`: List of solvers to use for comparison (default all solvers)
* `--use-symmetry`: Break symmetries in the presolve process.
* `--use-dual`: Use the dual model.
* `--use-no-presolve`: Avoid AMPL presolving process.

Finally, `compare_smt_encodings.py` presents the following parameter:
* `--solvers-list`: List of solvers to use for comparison (default *z3*)

## Repository structure

    .
    ├── images                              # Plots of the performances of different models for the different approaches
    │   ├── cp
    │   ├── lp
    │   ├── sat
    │   └── smt
    ├── instances                           
    │   ⋮
    │   └── ins-*-.txt                      # Instances to solve in `.txt` format
    ├── results                             # Json results of the performances of different models for the given approaches
    │   ├── cp
    │   ├── lp
    │   ├── sat
    │   └── smt
    ├── out                                 # Solutions for the given instances using different approaches
    │   ├── cp
    │   ├── cp-rotation
    │   ├── lp 
    │   ├── lp-rotation
    │   ├── sat
    │   ├── sat-rotation
    │   ├── smt
    │   └── smt-rotation
    ├── src
    │   ├── cp                      
    │   │   ├── data                        # Directory containing data examples for the problem in CP
    │   │   ├── models                      # Directory containing the models solving the problem in CP
    │   │   ├── rotation_models             # Directory containing the models solving the problem in CP considering rotations
    │   │   ├── solvers                     # Directory containing the solver configurations for CP
    │   │   ├── MODELS RECAP.md             # Recap of the CP MiniZinc models
    │   │   └── project_cp.mzp              # MiniZinc CP project
    │   ├── lp
    │   │   ⋮
    │   │   ├── model_*.mod                 # AMPL model solving the problem in LP
    │   │   ├── MODELS RECAP.md             # Recap of the LP AMPL models
    │   │   └── position_and_covering.py    # Script applying the Position and Covering technique for LP
    │   ├── sat
    │   │   ⋮
    │   │   ├── encoding_*.py               # Encoding solving the problem in LP
    │   │   ├── ENCODINGS RECAP.md          # Recap of the SAT encodings
    │   │   └── sat_utils.py                # Script containing useful functions for SAT
    │   ├── scripts                      
    │   │   ├── compare_cp_models.py        # Script to compare the results of CP models on the instances
    │   │   ├── compare_lp_models.py        # Script to compare the results of LP models on the instances
    │   │   ├── compare_sat_encodings.py    # Script to compare the results of SAT encodings on the instances
    │   │   ├── compare_smt_encodings.py    # Script to compare the results of SMT encodings on the instances
    │   │   ├── execute_cp.py               # Script to solve an instance using CP
    │   │   ├── execute_lp.py               # Script to solve an instance using LP
    │   │   ├── execute_sat.py              # Script to solve an instance using SAT
    │   │   ├── execute_smt.py              # Script to solve an instance using SMT
    │   │   ├── plot_comparisons.py         # Script to plot the results of the use of different models on the instances
    │   │   ├── solve_all_instances_cp.py   # Script solving all instances with CP
    │   │   ├── solve_all_instances_lp.py   # Script solving all instances with LP
    │   │   ├── solve_all_instances_sat.py  # Script solving all instances with SAT
    │   │   ├── solve_all_instances_smt.py  # Script solving all instances with SMT
    │   │   ├── solve_all_instances.py      # Script solving all instances with a desired methodology
    │   │   ├── unify_jsons.py
    │   │   ├── utils.py                    # Script containing useful functions
    │   │   └── visualize.py                # Script to visualize a solved instance
    │   └── smt
    │       ├── solvers                     # Directory containing the solvers executable files for SMT
    │       │   ⋮
    │       ├── encoding_*.py               # Encoding solving the problem in SMT
    │       ├── ENCODINGS RECAP.md          # Recap of the SMT encodings
    │       └── smt_utils.py                # Script containing useful functions for SMT
    ├── assignment.pdf                      # Assignment of the project
    ├── .gitattributes
    ├── .gitignore
    ├── LICENSE
    ├── report.pdf                          # Report of the project
    └── README.md

## Versioning

Git is used for versioning.

## Group members

|  Name           |  Surname  |     Email                           |    Username                                             |
| :-------------: | :-------: | :---------------------------------: | :-----------------------------------------------------: |
| Antonio         | Politano  | `antonio.politano2@studio.unibo.it` | [_S1082351_](https://github.com/S1082351)               |
| Enrico          | Pittini   | `enrico.pittini@studio.unibo.it`    | [_EnricoPittini_](https://github.com/EnricoPittini)     |
| Riccardo        | Spolaor   | `riccardo.spolaor@studio.unibo.it`  | [_RiccardoSpolaor_](https://github.com/RiccardoSpolaor) |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

