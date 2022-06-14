# Cardio Risk Calculator
Code repository for the paper [_Cardiac Operative Risk in Latin America: A Comparison of Machine Learning Models vs EuroSCORE-II_](https://www.sciencedirect.com/science/article/pii/S0003497521004483) published in The Annals of Thoracic Surgery

We present the first Machine Learning-based model for Cardiac Operative Risk in Latin America, you can try it out at [cardiorisk.ml](cardiorisk.ml).

### Disclaimer ###

The model was trained on Colombian Population from the city of Bogota and has been validated using retrospective data so far.

**IT IS NOT READY FOR MEDICAL USE**, and is provided as an educational tool only.


## Overview of the code

The code is structured as follows:

**Model Library**

* `cardio/models`: Contains implementations of different ML algorithms that follow the [Model](https://github.com/santiag0m/cardio/blob/db891e54ec88338aa14f49226ea6531ae59b9e39/cardio/models/model.py#L10) class.
* `cardio/utils.py`: Implements all the functions necessary for evaluation. It also contains Spanish to English translations of the features used. 

**Scripts**
* `benchmark.py`: Executes multiple train/evaluation experiments to compare evaluation metrics between all the ML algorithms, along with their confidence intervals. The outputs are saved in [JSON](https://www.json.org/json-en.html) format at: `data/outputs.json`
* `calibration_curves.py`: Re-runs experiment trials to calculate calibration curves for selected algorithms.
* `calibration_tables.py`: Calculates the calibration error of High Risk vs. Low Risk population subgroups for selected algorithms.
* `create_figures.py`: Reproduces the figures presented in the paper. Including: ROC and PR Curves, Confusion Matrices and SHAP interpretability scores.

## Results

The Gradient Boosting model, trained with our dataset of Colombian population, outperforms the established EuroSCORE-II in both the ROC and PR AUC metrics:

<p align="center">
  <img src="https://raw.githubusercontent.com/santiag0m/cardio/master/assets/roc_pr_results.jpg">
</p>

We also found (using the [SHAP](https://github.com/slundberg/shap) interpretability framework) that our model's risk assignment for certain variables is consistent with medical practice.
For example, healthy hematocrit ranges which should be between 41% - 50% (for men, according to the [American Red Cross](https://web.archive.org/web/20220508164156/https://www.redcrossblood.org/donate-blood/dlp/hematocrit.html)) 
are automatically identified by our moodel, as can be seen from its SHAP score:


<p align="center">
  <img width=512px src="https://raw.githubusercontent.com/santiag0m/cardio/master/assets/hematocrit_shap.PNG">
</p>

To see how other variables interact with our model, check the [Supplemental Material](https://ars.els-cdn.com/content/image/1-s2.0-S0003497521004483-mmc2.pdf)

## To Cite Us

```
@article{molina2022cardiac,
  title={Cardiac Operative Risk in Latin America: A Comparison of Machine Learning Models vs EuroSCORE-II},
  author={Molina, Ra{\'u}l Santiago and Molina-Rodr{\'\i}guez, Mar{\'\i}a Alejandra and Rinc{\'o}n, Francisco Mauricio and Maldonado, Javier Dario},
  journal={The Annals of Thoracic Surgery},
  volume={113},
  number={1},
  pages={92--99},
  year={2022},
  publisher={Elsevier}
}
```
