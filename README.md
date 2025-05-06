🧠 Stroke Prediction Using Machine Learning
Author: Smallman Jimu
Course/Department: Data Science
Project Type: / Final Project
Toolkits: Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn just to name a few


1. 📌 Introduction
Strokes are among the leading causes of death and disability worldwide. Early detection of individuals at risk of stroke is crucial to enable preventative interventions and reduce both mortality and long-term disability. In this project, machine learning algorithms were applied to a publicly available dataset to predict stroke occurrences based on health and demographic data. By exploring patterns in the data, the goal was to develop a predictive model that could support healthcare decision-making.


2. 🎯 Objective
The main objective of this project is to develop and evaluate classification models that can accurately predict whether an individual is likely to suffer a stroke. This was achieved by:

Exploring the stroke prediction dataset from Kaggle.

Preprocessing and cleaning the data.

Conducting exploratory data analysis to identify patterns and correlations.

Building and comparing multiple machine learning models.

Selecting the best-performing model.

Interpreting key features contributing to stroke predictions.

Identifying limitations and proposing future improvements.


3. 📂 Dataset Overview
The dataset used for this analysis is sourced from Kaggle, titled "Stroke Prediction Dataset". It includes 5,082 patient records with the following features:

Numerical Variables: age, avg_glucose_level, bmi.

Categorical Variables: gender, ever_married, work_type, Residence_type, smoking_status.

Binary Variables: hypertension, heart_disease.

Target Variable: stroke (1 = stroke occurred, 0 = no stroke)

Initial inspection revealed missing values (notably in the BMI column), and some categorical fields required encoding. The dataset presents a class imbalance challenge due to the rarity of stroke cases.


4. 🔧 Data Preprocessing
To prepare the dataset for machine learning, several preprocessing steps were carried out:

Handling Missing Values: Rows with missing BMI values were either dropped or imputed (depending on the model).

Categorical Encoding: Applied label encoding and one-hot encoding to convert categorical variables to numerical form.

Feature Scaling: Numerical features like age and glucose level were scaled using StandardScaler to normalize the range.

Train-Test Split: The dataset was split (typically 80/20) to evaluate model generalizability on unseen data.



5. 📊 Exploratory Data Analysis (EDA)
To better understand the structure and characteristics of the dataset: we also applied one hot encoding and visualised the categorical variabls indicating stroke and no stroke  converted to 0 and 1 thats the main use of one hot encording as it changes the categorical into numeric so it can be vsualised and read properly.

Correlation Matrix: A heatmap was generated to visualize correlations among numeric variables. by this correlation matrix the aim was to view the largest or least correlated variables in the dataset and as shown by the results that we got evr married and age with the highest of 0.70 and the least being -0.64 which is age and worktype.

Multicollinearity Check: VIF (Variance Inflation Factor) was computed for all predictors. All values were below 5, indicating no significant multicollinearity. The VIF measures the variance in each variable in the dataset and with the threshhold of 5 showing the maximum by which our model can be affected with variable with more variance but instead according to the findings it showed that the variance based on the lollipop plot did not exceed whch clearly shown that the variables used where good enough not to overlap.

Feature Importance: A random forest classifier was trained to rank feature importance. Results:

Average Glucose Level: Highest importance (~0.25) and more showing the highest contribution to stroke occurence which means that the more the glucose level the highest chance of an individual to have stroke.

BMI: Second most important (~0.23) this also like the glucose has the second largest effect according to the given variables to lead to stroke occurence

Age: Third (~0.21) thid largest as shown people with the age group of 60 and above leads to  the high stroke occurence a shown by the EDA analysis 

The 'id' feature was removed due to its irrelevance. it had no impact on the features given it was just redudant data so we had to remove it because it is insignificant enough not to include it on the stroke because it has zero input



6. 📉 Dimensionality Reduction with PCA
Principal Component Analysis (PCA) was employed to reduce dimensionality and visualize data variance.The components with the largest dataset and the elbow point was 4 showing the variables that contribute the most to the overal dataset, involving the bmi, glucose level and smoking which also shows tha variance ratio of the dataset variables.
An explained variance plot (elbow plot) was used to determine the optimal number of components to retain while minimizing information loss.
-PCA helped verify feature interdependence and potential redundancy but was not used in the final classification models as performance was higher with original features.


7. 🤖 Model Development and Evaluation

-Hyperparameters are the targets that was set with the random data and batches to test for the models which where going to be implemented and in this case it involved ,Logistic,SVC,Decision Trees and Random Forest and in the slides there is a diagram provided explaining the hyperparameter tuning through steps 1 to 4 which was training multiple models,evaluating them,Retraining again and evaluating through the accuracy and the f1 scores to look for  the best model overal and Logistic regression had the best hyperparameters which was proven by the code tself.

Four classification models were developed and tested:   

Logistic Regression

Support Vector Classifier (SVC)

Decision Tree Classifier

Random Forest Classifier

-ACTUAL DATA TESTING
Each model was trained using a Pipeline that included preprocessing steps. GridSearchCV was used for hyperparameter tuning. Model performance was evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Model Comparison:

Model	Accuracy	Notes
Logistic Regression	~0.78	Baseline performance
SVC	~0.87	Strong performer
Decision Tree	~0.85	Good but prone to overfitting
Random Forest	0.89	Best overall model



8. 📈 Confusion Matrix & Metrics
Each model's confusion matrix was analyzed to observe:

True Positives (correctly predicted strokes)

False Positives (stroke predicted but did not occur)

False Negatives (stroke occurred but not predicted)

True Negatives

The Random Forest model had the highest true positive rate and the most balanced precision-recall values, making it the most suitable model for deployment.



9. 🧠 Feature Importance Analysis
Using model-based and statistical techniques, the most important predictors for stroke were:

Age: Older age increased stroke risk.

Average Glucose Level: Elevated glucose levels were a significant risk factor.

BMI: Obesity correlated with higher stroke probability.

Hypertension: Associated with a higher chance of stroke.

Smoking Status: "Smokes" category showed positive correlation.

Statistical Significance: All top features had p-values < 0.05.



10. 🌳 Decision Tree Insights
The decision tree model provided visual interpretability:

Root Node: Age ≤ 0.012 split the data.

Important Splits: Age, work_type, and smoking_status frequently appeared near the top levels.

Although not the best-performing model in terms of accuracy, the decision tree offered valuable insights into the feature decision flow.



11. ⚠️ Limitations
Despite achieving promising results, several limitations were observed:

Class Imbalance: The small number of stroke cases skewed model predictions.

Limited Variables: Key health data (e.g., diet, medication use) was not included.

Overfitting Risk: Tree-based models showed signs of overfitting without regularization.

Dataset Size: Only ~5,000 records limited model generalization.

Lack of Real-world Testing: No external validation was conducted using real clinical data.



12. 🔮 Future Work & Solutions to Limitations
Comprehensive future strategies to address limitations include:

Feature enhancement by integrating data from health records or wearable devices.

Model regularization (e.g., pruning, limiting tree depth).

Explainability tools (e.g., SHAP, LIME) for black-box models.

Testing across healthcare settings for better generalizability.

Ethical practices and data privacy adherence during data collection and deployment.

➡️ See the detailed “Future Work and Solutions” section above for full strategies.

13. ✅ Conclusion
The stroke prediction project demonstrates how machine learning can be applied to health data to support early diagnosis and prevention strategies. The Random Forest model emerged as the most effective classifier, showing robust predictive performance and interpretability. With further improvements in data quality and volume, this approach could significantly aid clinicians in identifying at-risk patients proactively.


--------------------THANK YOU