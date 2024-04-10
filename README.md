Our **datasets** directory structure:
* **original_datasets**: .tsv datasets directly from the publication
  * B3DB_regression.tsv
  * B3DB_classification.tsv
* **cleaned_datasets**: quick cleaned up datasets that only keep the SMILES 
  and property to predict into .csv 
  * BBB_regression.csv
  * BBB_classification.csv
* **expanded_datasets**: .csv.zip datasets that have been expanded by RDKit descriptors, Morgan fingerprints, and MACCS keys
  * BBB_regression_expanded.csv.zip <- This is the final dataset for 
    regression models
  * BBB_classification_expanded.csv.zip
* **balanced_datasets**: These are the final datasets for 
  classification models
  * BBB_classification_balanced_centroid.csv.zip
  * BBB_classification_balanced_smoteenn.csv.zip 
* **holdout_datasets**: After cleaning, expanding, and balancing the datasets, they were divided into two groups. This dataset was reserved for post-training validation.
  * classification_df_expanded_cleaned_holdout.csv.zip
  * regression_df_expanded_cleaned_holdout.csv.zip
* **train_datasets**: These are the datasets used for training after the division.
  * classification_df_expanded

Our **model_outputs** directory structure:
* **data_preprocessing**: Graphic representations of our data, as well as the effects of some processing.
  * regression_histrogram.png
  * classification_classes_counts.png
  * classification_before_balancing.png
  * classification_balanced_centroid.png
  * classification_balanced_smoteenn.png
* **mlp_classifier**:
* **rf_regressor**:
* **svm_classifier**: These are pickles(saved data and objects) and graphics from the SVM training and evaluation, including test sets, PCA components, and processing tools.
  * Data splits
    * centroid_x_train.pkl
    * centroid_y_train.pkl
    * centroid_x_test.pkl
    * centroid_y_test.pkl
    * smoteenn_x_train.pkl
    * smoteenn_y_train.pkl
    * smoteenn_x_test.pkl
    * smoteenn_y_test.pkl
  * PCA Components
    * top_two_centroid.pkl
    * top_two_smoteenn.pkl
  * Pipelines
    * centroid_pipeline.pkl
    * smoteenn_pipeline.pkl
  * Evaluation Graphics
    * centroid_auroc.png
    * centroid_holdout_roc.png
    * centroid_metrics.png
    * smoteenn_auroc.png
    * smoteenn_holdout_roc.png
    * SMOTEENN_metrics.png
* **svr_regressor**:These are pickles(saved data and objects) and graphics from the regressor training and evaluation, including test sets, PCA components, and processing tools.
  * Data splits
    * regression_x_train.pkl
    * regression_x_test.pkl
    * regression_y_train.pkl
    * regression_y_test.pkl
  * Pipelines
    * svm_regressor_pipeline.pkl
  * Evaluation Graphics
    * metrics.png
    * pred_vs_actual.png
    * residuals.png

Our **model_pickles** directory contains the trained models fit with the best parameters, as pickles:
* best_mlp_classifier.pkl
* best_rf_regressor.pkl
* best_svm_classifier_centroid.pkl
* best_svm_classifier_smoteenn.pkl
* best_svm_regressor.pkl

**Jupyter Notebooks**

data_preprocessing.ipynb : This is where the datasets were cleaned, balanced, and a holdout set was separated out

baseline_models_training.ipynb : The SVM and SVR models were trained here using a grid search
baseline_models_evaluation.ipynb : The SVM and SVR models were evaluated using residuals, ROC, Delongs, etc.

complex_models_training.ipynb : The Random Forest and Multilayer Perceptron were trained here using a grid search
complex_models_evaluation.ipynb : The RF and MLP were evaluated using various validations, such as ROC and residuals.


**Python Files**

dataset_cleanup.py :
dataset_expansion.py :
dataset_plot.py : 

model_evaluation.py : functions used in evaluating models, notably "Delong's Test"
pickle_managment.py : functions used to help save and store models and other objects for future use



