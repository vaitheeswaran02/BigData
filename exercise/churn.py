5. Practical implementation using PySpark
Practical implementation
Now we'll implement the complete Customer Churn Prediction project.
Tools required
- Ubuntu Linux
- Python 3
- Apache Spark / PySpark
- Jupyter Notebook or VS Code (optional)
- Customer churn CSV dataset
We'll use Logistic Regression as our Machine Learning algorithm.
Step 1: Verify PySpark installation
You previously installed PySpark 4.2.0 in your virtual environment, so let's verify that setup.

Activate your existing virtual environment:
source /path/to/spark-env/bin/activate


Replace the path with the actual location of your spark-env directory.

Check Python:
python --version


Check PySpark:
python -c "import pyspark; print(pyspark.__version__)"


Check Java:
java -version


Step 2: Download the customer churn dataset
For this practical, use the IBM Telco Customer Churn dataset.

Dataset filename:
WA_Fn-UseC_-Telco-Customer-Churn.csv


You can find it here:
Download IBM Telco Customer Churn dataset – KaggleThe dataset contains approximately 7,043 customer records.
Important dataset columns
Column	Description
customerID	Unique customer identifier
gender	Customer gender
SeniorCitizen	Senior citizen indicator
Partner	Whether customer has a partner
Dependents	Whether customer has dependents
tenure	Number of months as a customer
PhoneService	Phone service availability
InternetService	Internet service type
Contract	Contract type
MonthlyCharges	Monthly charges
TotalCharges	Total charges
Churn	Whether the customer left

Our target column is Churn.
Step 3: Create a project directory
Open your terminal:
mkdir -p ~/BigData/CustomerChurn
cd ~/BigData/CustomerChurn


Place the downloaded CSV file inside this directory.

Verify:
ls


Expected:
WA_Fn-UseC_-Telco-Customer-Churn.csv


Step 4: Create the PySpark program
Create a Python file:
nano churn_prediction.py


We will divide the program into six sections.
Section A: Import libraries and initialize Spark
from pyspark.sql import SparkSessionfrom pyspark.sql.functions import colfrom pyspark.ml import Pipelinefrom pyspark.ml.feature import (    StringIndexer,    OneHotEncoder,    VectorAssembler,    Imputer)from pyspark.ml.classification import LogisticRegressionfrom pyspark.ml.evaluation import (    BinaryClassificationEvaluator,    MulticlassClassificationEvaluator)# Initialize Sparkspark = SparkSession.builder \    .appName("CustomerChurnPrediction") \    .master("local[*]") \    .getOrCreate()spark.sparkContext.setLogLevel("WARN")print("Spark initialized successfully")



Explanation:

We import:
- SparkSession to initialize Spark.
- Pipeline to organize preprocessing and model training.
- StringIndexer and OneHotEncoder to transform categorical data.
- VectorAssembler to combine features.
- LogisticRegression to build our prediction model.
- Evaluators to measure model performance.
Section B: Load the dataset
# Load CSV datasetdf = spark.read.csv(    "WA_Fn-UseC_-Telco-Customer-Churn.csv",    header=True,    inferSchema=True)# Display first 5 recordsdf.show(5)# Display schemadf.printSchema()# Display total recordsprint("Total records:", df.count())



Explanation:
- header=True: First row contains column names.
- inferSchema=True: Spark attempts to identify column data types.
- show(5): Displays five records.
- printSchema(): Displays column names and data types.
- count(): Counts records.
Section C: Data preprocessing
This is one of the most important steps.

The original dataset contains some empty values in TotalCharges.

We also need to remove unnecessary columns.
# Remove customerIDdf = df.drop("customerID")# Convert TotalCharges into numeric formatdf = df.withColumn(    "TotalCharges",    col("TotalCharges").cast("double"))# Remove duplicate recordsdf = df.dropDuplicates()# Display missing valuesdf.select([    col(c).isNull().cast("int").alias(c)    for c in df.columns]).groupBy().sum().show()print("Preprocessing completed")



Why do we remove customerID?

CustomerID is just an identifier. It does not meaningfully describe customer behaviour for this prediction task.

Why convert TotalCharges?

The original column can be inferred as a string because some records contain blank values. Logistic Regression requires numerical input features.
Section D: Feature engineering
Feature engineering converts raw customer information into numerical features that a Machine Learning model can understand.

First, identify numerical and categorical columns.
# Numerical columnsnumeric_cols = [    "tenure",    "MonthlyCharges",    "TotalCharges"]# Categorical columnscategorical_cols = [    "gender",    "Partner",    "Dependents",    "PhoneService",    "MultipleLines",    "InternetService",    "OnlineSecurity",    "OnlineBackup",    "DeviceProtection",    "TechSupport",    "StreamingTV",    "StreamingMovies",    "Contract",    "PaperlessBilling",    "PaymentMethod"]



Now handle missing numerical values using Imputer.
imputer = Imputer(    inputCols=numeric_cols,    outputCols=[c + "_imputed" for c in numeric_cols],    strategy="median")



Convert categorical columns into numerical indexes:
indexer = StringIndexer(    inputCols=categorical_cols,    outputCols=[c + "_index" for c in categorical_cols],    handleInvalid="keep")



Convert categorical indexes into binary feature vectors:
encoder = OneHotEncoder(    inputCols=[c + "_index" for c in categorical_cols],    outputCols=[c + "_encoded" for c in categorical_cols],    handleInvalid="keep")



Convert the target variable:
label_indexer = StringIndexer(    inputCol="Churn",    outputCol="label",    handleInvalid="error")



Combine all features:
assembler = VectorAssembler(    inputCols=(        [c + "_imputed" for c in numeric_cols] +        [c + "_encoded" for c in categorical_cols]    ),    outputCol="features",    handleInvalid="error")



What is the meaning of the features column?

It combines all numerical and encoded categorical values into one vector.

Example:
features = [12.0, 70.5, 850.0, 1.0, 0.0, 1.0, ...]


The model uses this vector to make predictions.
Section E: Split the dataset and train the model
First, divide the dataset:
train_data, test_data = df.randomSplit(    [0.8, 0.2],    seed=42)print("Training records:", train_data.count())print("Testing records:", test_data.count())



Create the Logistic Regression model:
lr = LogisticRegression(    featuresCol="features",    labelCol="label",    maxIter=50)



Create a Machine Learning pipeline:
pipeline = Pipeline(stages=[    imputer,    indexer,    encoder,    label_indexer,    assembler,    lr])



Train the model:
model = pipeline.fit(train_data)print("Model training completed")



What is a Pipeline?

A Pipeline combines multiple processing stages into one workflow.

Our pipeline performs:
Raw Data
   |
   v
Missing Value Handling
   |
   v
Categorical Encoding
   |
   v
Feature Assembly
   |
   v
Logistic Regression
   |
   v
Trained Model


An important advantage is that preprocessing stages are fitted using training data only. The same fitted transformations are then applied to testing data, preventing preprocessing leakage.
Section F: Predict customer churn
Now use the trained model on the test dataset.
predictions = model.transform(test_data)predictions.select(    "label",    "prediction",    "probability").show(10, truncate=False)



Here:
- label = Actual result.
- prediction = Predicted result.
- probability = Probability distribution for each class.
For example:
Actual     Predicted
1.0        1.0
0.0        0.0
1.0        0.0
0.0        1.0


Where:
- 1.0 means churn.
- 0.0 means no churn.
The label mapping is based on StringIndexer; in this dataset, Yes will normally map to 1 and No to 0. You can verify the mapping from label_indexer_model.labels.

6. Evaluate the Machine Learning model
After prediction, we need to measure the model's performance.
A. Accuracy
evaluator = MulticlassClassificationEvaluator(    labelCol="label",    predictionCol="prediction",    metricName="accuracy")accuracy = evaluator.evaluate(predictions)print("Accuracy:", accuracy * 100, "%")



B. Precision
precision_evaluator = MulticlassClassificationEvaluator(    labelCol="label",    predictionCol="prediction",    metricName="weightedPrecision")print("Precision:", precision_evaluator.evaluate(predictions))



C. Recall
recall_evaluator = MulticlassClassificationEvaluator(    labelCol="label",    predictionCol="prediction",    metricName="weightedRecall")print("Recall:", recall_evaluator.evaluate(predictions))



D. F1-score
f1_evaluator = MulticlassClassificationEvaluator(    labelCol="label",    predictionCol="prediction",    metricName="f1")print("F1 Score:", f1_evaluator.evaluate(predictions))



E. AUC score
auc_evaluator = BinaryClassificationEvaluator(    labelCol="label",    rawPredictionCol="rawPrediction",    metricName="areaUnderROC")print("AUC:", auc_evaluator.evaluate(predictions))



F. Confusion matrix
predictions.groupBy(    "label",    "prediction").count().show()



This helps identify:
- True Positive
- True Negative
- False Positive
- False Negative
Note: Accuracy alone is not sufficient for churn prediction because the dataset has more non-churn customers than churn customers. Pay particular attention to churn-class precision and recall when evaluating the model.
7. Complete execution
Save your Python file:
Ctrl + O
Enter
Ctrl + X


Run the program:
python churn_prediction.py


Alternatively:
spark-submit churn_prediction.py


At the end of the program, add:
spark.stop()



This stops the SparkSession and releases its resources.
Expected output
Your output should contain:
1. Spark initialization successful.
2. Dataset records and schema.
3. Total number of records.
4. Missing-value information.
5. Training and testing record counts.
6. Model training completed.
7. Customer churn predictions.
8. Accuracy, precision, recall, F1-score and AUC.
Example output format

Illustrative output only. Actual metrics depend on the dataset and train-test split.
Spark initialized successfully

Total records: 7043

Training records: 5630
Testing records: 1413

Model training completed

Accuracy: 80.5 %
Precision: 0.79
Recall: 0.80
F1 Score: 0.79
AUC: 0.84
