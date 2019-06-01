## Analysis notebooks and codes for benchmarking project.

This repository contains notebooks or scripts used to perform analysis for regression benchmarking manuscript.

## Manuscript

__Title: Benchmarking the predictive performance of regression methods for microbiome data__

__Abstract__

__Background:__  Large-scale microbiome datasets from 16S rRNA gene amplicon sequencing have provided an opportunity to build predictive models using supervised machine learning algorithms. Regression analyses have shown the predictive potential to link microbial indicators to features of the microbial environment, such as soil pH, postmortem interval, and infant age. However,  little justification is typically given for the use of specific methods for characteristically sparse, compositional microbiome data with high feature counts, and almost no information is available about the time and processing power needed to generate and optimize these models. 

__Results:__ We benchmarked the performance and computational requirements for 36 regression methods across 5 families, using datasets from 20 separate microbiome studies ranging from 11 to 1160 samples. For each method, we provide performance indicators such as predictive accuracy, runtime, and sensitivity to parameter selection. Ensemble methods (such as Random Forest, extremely randomized trees, and gradient boosting) demonstrated the highest overall performance for regression tasks. We observed a tradeoff between performance and run time, though the impact of hyperparameter tuning efficiency varied across the methods. Among ensemble methods, Random Forest-based models have smaller hyperparameter spaces and are the most efficient approach, achieving the best prediction accuracy within reasonable run times.

__Conclusions:__ For regression analyses with 16S rRNA gene amplicon data, Random Forest-based models outperform other available models in  accuracy and runtime. We expect these conclusions to also apply to compositional data obtained from shotgun metagenomic analyses.

