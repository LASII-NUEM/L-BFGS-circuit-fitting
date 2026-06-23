# A quasi-Newton approach to gas-hydrate-oriented circuit fitting problems

## Abstract
Gas hydrates are among the greatest concerns within deep-sea petroleum extraction. Their fast growth rates, especially under favorable conditions such as high pressure and low temperature, can lead to severe pipe clogging and costly stoppages. However, hydrate monitoring in the oil industry is usually limited to temperature and pressure readings. These techniques detect the presence of the structures but lead to few insights into physical and electrical properties. As a solution to this limitation, impedance spectroscopy has been reported to accurately measure hydrate dynamics and, more importantly, at a fairly low cost. Impedance readings across the broad frequency range inherent to this method enable direct characterization of multiple hydrate-related phenomena. For instance, parameters such as volume, saturation, and salt levels can be derived through the frequency response. Furthermore, from the impedance measurements, it is possible to define an inverse problem to identify an approximate electrical circuit that generated said output. This task is known as electric circuit fitting. This paper proposes using a quasi-Newton minimizer as an alternative to usual estimators in the literature, targeting hydrate-oriented spectra. Four samples were selected, spanning different frequency ranges and acquisition hardware, to assess the method's viability across multiple scenarios. Results show that the proposed approach outperforms the most common algorithms for circuit-fitting problems, achieving improved fit scores and faster convergence. Additionally, this work provides an accurate framework for future efforts on hydrate-oriented electrical characterizations.

## Repository configuration
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

## Datasets
This repository provides the raw data for the four different hydrate-oriented spectra selected for the experimental data:
- [`Methane hydrates`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/data/methane_hydrate): samples synthesized from seawater (with and without sand), extracted from the open dataset published in [10.1029/2020GL087645](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL087645). Acquisition hardware: GwINSTEK LCR 6300 meter;
- [`THF hydrate`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/data/tetrahydrofuran_hydrate): Hydrate formed from a THF-water mixture placed in a 16452A test fixture and frozen in a thermal bath for 40 minutes. Acquisition hardware: 4294A impedance analyzer;
- [`Ice from distilled water`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/data/ice/ice_distilled_E4980_LCR): Distilled water placed in a 16452A test fixture and frozen in a commercial freezer for approximately 3 hours. Acquisition hardware:  Keysight E4980 LCR meter.

## Reproducing the paper results
All the results presented in the paper can be reproduced by the scripts provided in the [`results`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/results) directory. For each sample, an individual script is provided that can be run independently. The only infrastructure requirement is to have the [`framework`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/framework) recognized by the scripts and the raw data in an accessible directory (e.g., [`data`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/tree/main/data) at the root). Except for [`time_benchmark.py`](https://github.com/LASII-NUEM/L-BFGS_circuit_fitting/blob/main/results/time_benchmarks.py), which reproduces the results in Table II, the other scripts pertain to Fig. 2 and Table I of the paper. With the environment activated, run each script in the terminal with:
```
python3 ./<result_script_name>.py 
```
assuming the script is called from the `results` directory. Alternatively, run the script directly using the IDE's run command.

[//]: # (## Fit it yourself!)


## Related works
- **M. Y. Müller, E. Trento Jr., J. P. N. Longo, I. C. Santos, M. Marcelino Neto, R. Morales, T. A. R. Passarin and E. N. Santos**, ''A quasi-Newton approach to gas-hydrate-oriented circuit fitting problems''. 19th International Workshop on Impedance Spectroscopy, 2026 - submitted
- **E. Trento Jr., M. Y. Müller, J. P. N. Longo, M. Marcelino Neto, R. Morales and E. N. Santos**. "Development of a multi-electrode sensor for hydrate characterization: An ice formation case study". 25th IEEE Conference on Sensors, 2026 - submitted

## L-BFGS_circuit_fitting collaborators
- [Mateus Yamada Müller](http://lattes.cnpq.br/9256538421307823)
- [Everton Trento Junior](http://lattes.cnpq.br/8125264101828725)
- [Jean P. Nakatu Longo](http://lattes.cnpq.br/1296823996110131)
- [Igor Centurião dos Santos](http://lattes.cnpq.br/4585976716721109)
- [Thiago Alberto Rigo Passarin](http://lattes.cnpq.br/1366903848418887)
- [Eduardo Nunes dos Santos](http://lattes.cnpq.br/1324570975148136)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
