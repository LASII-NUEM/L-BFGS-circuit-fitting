# A quasi-Newton approach to gas-hydrate-oriented circuit fitting problems

# Abstract
Gas hydrates are among the greatest concerns within deep-sea petroleum extraction. Their fast growth rates, especially under favorable conditions such as high pressure and low temperature, can lead to severe pipe clogging and costly stoppages. However, hydrate monitoring in the oil industry is usually limited to temperature and pressure readings. These techniques detect the presence of the structures but lead to few insights into physical and electrical properties. As a solution to this limitation, impedance spectroscopy has been reported to accurately measure hydrate dynamics and, more importantly, at a fairly low cost. Impedance readings across the broad frequency range inherent to this method enable direct characterization of multiple hydrate-related phenomena. For instance, parameters such as volume, saturation, and salt levels can be derived through the frequency response. Furthermore, from the impedance measurements, it is possible to define an inverse problem to identify an approximate electrical circuit that generated said output. This task is known as electric circuit fitting. This paper proposes using a quasi-Newton minimizer as an alternative to usual estimators in the literature, targeting hydrate-oriented spectra. Four samples were selected, spanning different frequency ranges and acquisition hardware, to assess the method's viability across multiple scenarios. Results show that the proposed approach outperforms the most common algorithms for circuit-fitting problems, achieving improved fit scores and faster convergence. Additionally, this work provides an accurate framework for future efforts on hydrate-oriented electrical characterizations.

# Table of Contents
1. [Repository configuration](#1-repository-configuration)
2. [Datasets](#2-datasets)
3. [Reproducing the paper results](#3-reproducing-the-paper-results)
4. [Fit it yourself!](#4-fit-it-yourself)
5. [Related works](#5-related-works)
6. [Repository collaborators](#6-repository-collaborators)
7. [License](#7-license)
8. [Contributing](#8-contributing)
9. [Questions?](#9-questions)

# 1. Repository configuration 
To set up the toolbox, first clone this repository on your machine:

```
cd <desired_path>
git clone https://github.com/LASII-NUEM/L-BFGS_circuit_fitting.git
cd L-BFGS_circuit_fitting
```

Then, configure a Python environment. The framework has been successfully tested in Python 3.12 and 3.13. Also, in both Linux and Windows operating systems. Lastly, install the required packages listed in the requirements text file at the root of the repository with:

```
pip install -r requirements.txt
```

Once all packages are installed, the toolbox is ready for use. In the following sections, you may find some examples of how to reproduce our circuit fitting results.

# 2. Datasets 
This repository provides the raw data for the four different hydrate-oriented spectra selected for the experimental data:
- [`Methane hydrates`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/data/methane_hydrate): samples synthesized from seawater (with and without sand), extracted from the open dataset published in [10.1029/2020GL087645](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL087645). Acquisition hardware: GwINSTEK LCR 6300 meter;
- [`THF hydrate`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/data/tetrahydrofuran_hydrate): Hydrate formed from a THF-water mixture placed in a 16452A test fixture and frozen in a thermal bath for 40 minutes. Acquisition hardware: 4294A impedance analyzer;
- [`Ice from distilled water`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/data/ice/ice_distilled_E4980_LCR): Distilled water placed in a 16452A test fixture and frozen in a commercial freezer for approximately 3 hours. Acquisition hardware:  Keysight E4980 LCR meter.

# 3. Reproducing the paper results 
All the results presented in the paper can be reproduced by the scripts provided in the [`results`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/results) directory. For each sample, an individual script is provided that can be run independently. The only infrastructure requirement is to have the [`framework`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/framework) recognized by the scripts and the raw data in an accessible directory (e.g., [`data`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/data) at the root). Except for [`time_benchmark.py`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/blob/main/results/time_benchmarks.py), which reproduces the results in Table II, the other scripts pertain to Fig. 2 and Table I of the paper. With the environment activated, run each script in the terminal with:
```
python3 ./<result_script_name>.py 
```
assuming the script is called from the `results` directory. Alternatively, run the script directly using the IDE's run command.

# 4. Fit it yourself! 

In case the user would like to test the L-BFGS algorithm for fitting their own data, [`L-BFGS_showcase.py`](https://github.com/LASII-NUEM/L-BFGS-circuit-fitting/blob/main/L-BFGS_showcase.py) provides the required steps to do so. For the fitting task to work, two steps are required: **defining an ECM function** and **loading the impedance spectroscopy data**. 

In [`equivalent_circuits.py`](https://github.com/LASII-NUEM/L-BFGS-circuit-fitting/blob/main/framework/equivalent_circuits.py), multiple hydrate-relevant ECMs are already implemented, and we first recommend testing one out before implementing your own. Nonetheless, adding one from scratch is straightforward. Make sure the input arguments are the same as those from the other existing functions, and that the function returns the frequency response. Also, update the "function_handlers" dictionary in [`fitting_utils.py`](https://github.com/LASII-NUEM/PHOBOS/blob/main/framework/fitting_utils.py) with the required parameters for your new model. 

As for impedance spectroscopy data, it can be passed either as framework-native data types (e.g., SpectroscopyData from [`data_types`](https://github.com/LASII-NUEM/L-BFGS-circuit-fitting/blob/main/framework/data_types.py)) or as a list containing both the real and imaginary parts of the impedance. Native data types require that their raw CSV data comply with predefined formats. Please see [`showcase_files`](https://github.com/LASII-NUEM/L-BFGS-circuit-fitting/tree/main/data/showcase) for more information. Regardless, libraries such as [pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html), [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html), or built-in functions including [open](https://docs.python.org/3/library/functions.html#open) can handle most common file formats.

Lastly, once the target data has been loaded, create an object of the "EquivalentCircuit" class and run the fitting task:
```
from framework import fitting_utils

model = <name_of_the_model> #key to the model parameters in function_handlers dictionary 
freqs = <frequencies> #array with the swept frequencies in Hz 
obj = fitting_utils.EquivalentCircuit(model, <impedance_data>, freqs) #equivalent circuit object
fit_params_BFGS = fit_obj.fit_circuit(<initial_guess>, <scaling_array>, method="BFGS", verbose=True) #L-BFGS fit routine
```

The `fit_circuit` method returns an object of the [OptimizerResults](https://github.com/LASII-NUEM/L-BFGS-circuit-fitting/blob/main/framework/optimization_utils.py) class containing fit statistics, such as optimal parameters, cost function values, error metrics, and computation time.

# 5. Related works 
- **M. Y. Müller, E. Trento Jr., J. P. N. Longo, I. C. Santos, M. Marcelino Neto, R. E. M. Morales, T. A. R. Passarin and E. N. Santos**, ''A quasi-Newton approach to gas-hydrate-oriented circuit fitting problems''. 19th International Workshop on Impedance Spectroscopy, 2026 - submitted
- **E. Trento Jr., M. Y. Müller, J. P. N. Longo, M. Marcelino Neto, R. Morales and E. N. Santos**. "Development of a multi-electrode sensor for hydrate characterization: An ice formation case study". 25th IEEE Conference on Sensors, 2026 - submitted

# 6. Repository collaborators 
- [Mateus Yamada Müller](http://lattes.cnpq.br/9256538421307823)
- [Everton Trento Junior](http://lattes.cnpq.br/8125264101828725)
- [Jean P. Nakatu Longo](http://lattes.cnpq.br/1296823996110131)
- [Igor Centurião dos Santos](http://lattes.cnpq.br/4585976716721109)
- [Moises Marcelino Neto](http://lattes.cnpq.br/2071333457212415)
- [Rigoberto Eleazar Melgarejo Morales](http://lattes.cnpq.br/5156573817670917)
- [Thiago Alberto Rigo Passarin](http://lattes.cnpq.br/1366903848418887)
- [Eduardo Nunes dos Santos](http://lattes.cnpq.br/1324570975148136)

# 7. License
All Python source code (.py) is made available under the MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See LICENSE-MIT.txt for the full license text.

# 8. Contributing 
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# 9. Questions? 
If you have any questions, please feel free to reach out via email at mateusyamada@alunos.utfpr.edu.br!
