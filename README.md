# EnContact
EnContact: predicting enhancer-enhancer contacts using sequence-based deep learning model

# Model training
We collected chromatin contact matrix of seven cell lines (i.e. GM, K562, HCASMC, MyLa, Naïve, Th17, and Treg) from HiChIP data of Mumbach et al, 2017. We implemented the EnContact model using Keras 1.2.0 on a Linux server. All experiments were carried out with 4 Nvidia K80 GPUs which significantly accelerated the training process than CPUs. 

# Application: predict enhancer-enhancer interactions from HiChIP data
We apply the trained EnContact model to infer contacts between enhancers in situations where one or both interaction regions contain multiple regulatory elements. In this way, we predict enhancer-enhancer interactions from bin-level interactions. For each cell line, the predicted interactions are saved in 'predictions_cell.txt' files (e.g. EnContact/E-E prediction/predictions_GM.txt).

# Requirements
- hickle
- numpy=1.13.3
- Theano=0.8.0
- keras=1.2.0
- pandas=0.20.1
- biopython=1.70
- Scikit-learn=0.18.2

# Installation
Download EnContact by
```shell
git clone https://github.com/liwenran/DeepTACT
```

# License
This project is licensed under the MIT License - see the LICENSE.md file for details
