# Heart Disease Prediction

## Project Overview

This project focuses on predicting the presence of heart disease using **supervised machine learning algorithms**. It addresses a binary classification problem based on clinical features. As my first machine learning project, it provided hands-on experience with the end-to-end workflow of building and evaluating predictive models.

## Features
- Perform Exploratory Data Analysis (EDA) to uncover relationships between clinical features and target variable, including visualization and statistical summaries.
- Implemented and compared multiple supervised machine learning algorithms for binary classification:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- Naive Bayes Classifier

## Workflow Employed:
The project follows a strucuture machine learning pipeline:

1.**Import Libraries**
load essential python libraries as `pandas`, `numpy`, `matplotlib`, `seaborn` and `scikit-learn` for data handling, visualization and modeling.

2.**Load Dataset**
Read the dataset into a Pandas DataFrame and inspect its structure using `.head()`, `.info()`, and `describe()` to understand feature types and distributions.

3.**Exploratory Data Analysis (EDA)**
- Visualize feature distribution and relationships using histograms, bar plots, and pair plots.
- Analyze correlations using heatmaps.
- Investigate patterns between features and the target variable.

4.**Data Preprocessing**
- Handling missing values
- Identify categorical and numerical features
- Define features and target
- Split the dataset into training and testing dataset using `train_test_split()`
- Encode categorical features
- Normalize numerical features using `StandardScaler`

5.**Modeling**
- Insantiate models: Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes.
- Fit models on the training data using `.fit()`

6.**Evaluation**
-Predict on test data using `.predict()`
- Evaluate using metrics: `accuracy_score`, `confusion_matrix`, and `classification_report`.
- Compare model performance to identify the most effective model.

## Results:
The performance of each supervised machine learning algorithm was evaluted using accuracy, precision, recall and F1-score. Below is a summary of the results:

+---------------------------------------------------------------------+
| Model                    | Accuracy | Precision | Recall | F1-Score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression      |  86.89%  |  0.8689   | 0.8689 |  0.8689  |
| Decision Tree Classifier |  85.25%  |  0.8633   | 0.8525 |  0.8521  |
| Random Forest Classifier |  83.61%  |  0.8361   | 0.8361 |  0.8361  |
| Support Vecotr Machine   |**88.52%**|  0.8855   | 0.8852 |  0.8851  |
| Naive Bayes Classifier   |  86.89%  |  0.8709   | 0.8689 |  0.8689  |
+---------------------------------------------------------------------+

The **Support Vector Machine** achived the highest overall performance, making it the most effective model fot this classifiaction task.

## Tools and Libraries
- Python(NumPy, Pandas)
- Scikit-learn
- Matplotlib & Seaborn
- Jupyter Notebook

## How to run
1. Clone the repository:
   ```bash
   git colne <https://github.com/username/project-name.git>
   cd <project_name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook
   ```bash
   jupyter notebook
   ```

## Project Structure
Heart_Disease_Prediction/
|
|-- Heart_Disease_Prediction.ipynb # Main Jupyter Notebook
|-- HeatDisease.xls                # Dataset file
|-- README.md                      # Project overview and documentation
|-- requirements.txt               # Python dependencies

## Author

**Akshat Girish Dandur**
B.Tech in Aritificial Intelligence & Machine Learning
Passionate about applying machine learning to real-world problem in healthcare, electric mobility, and beyond.

- GitHub: https://github.com/Akshat-GD
- LinkedIn: https://www.linkedin.com/in/akshat-dandur-a3817932a/
- Email: akshatdandur@gmail.com

*Feel free to reach out for collaboration, feedback, or just to talk tech!*