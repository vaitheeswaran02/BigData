Step 1: Install the required Python packages
sudo apt update
sudo apt install python3-full python3-venv

step2: installl num py 
python -m pip install numpy
Verify the installation
python -c "import numpy; print(numpy.__version__)"

Step 2: Create a virtual environment
mkdir -p ~/spark-exercise11
cd ~/spark-exercise11
python3 -m venv spark-env

Step 3: Activate the environment
source spark-env/bin/activate

Step 4: Install PySpark
Now run:
python -m pip install --upgrade pip
python -m pip install pyspark

Step 5: Verify the installation
python -c "import pyspark; print(pyspark.__version__)"
====================================================================

Step 1: Create the file
Run this in your terminal. Make sure your virtual environment is activated.
cd ~/spark-exercise11
source spark-env/bin/activate
nano churn_prediction.py


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create Spark session
spark = SparkSession.builder \
    .appName("CustomerChurnPrediction") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Historical customer dataset
# churn: 0 = Customer stays, 1 = Customer leaves
data = [
    (22, 2, 80.0, 160.0, 1),
    (25, 4, 75.0, 300.0, 1),
    (28, 6, 70.0, 420.0, 0),
    (35, 24, 55.0, 1320.0, 0),
    (40, 36, 60.0, 2160.0, 0),
    (45, 3, 90.0, 270.0, 1),
    (32, 18, 65.0, 1170.0, 0),
    (29, 5, 85.0, 425.0, 1),
    (50, 48, 50.0, 2400.0, 0),
    (23, 1, 95.0, 95.0, 1),
    (38, 30, 58.0, 1740.0, 0),
    (42, 8, 78.0, 624.0, 1),
    (31, 20, 62.0, 1240.0, 0),
    (27, 3, 88.0, 264.0, 1),
    (55, 60, 45.0, 2700.0, 0),
    (36, 9, 82.0, 738.0, 1),
    (48, 42, 52.0, 2184.0, 0),
    (24, 2, 92.0, 184.0, 1),
    (39, 28, 57.0, 1596.0, 0),
    (33, 7, 76.0, 532.0, 1)
]

columns = [
    "age",
    "tenure",
    "monthly_charges",
    "total_charges",
    "churn"
]

df = spark.createDataFrame(data, columns)

print("\n=== Historical Customer Dataset ===")
df.show()

# Prepare input features for machine learning
assembler = VectorAssembler(
    inputCols=[
        "age",
        "tenure",
        "monthly_charges",
        "total_charges"
    ],
    outputCol="features"
)

dataset = assembler.transform(df).select(
    "features",
    df["churn"].alias("label")
)

# Split dataset into training and testing data
train, test = dataset.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training records:", train.count())
print("Testing records:", test.count())

# Create Logistic Regression model
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50
)

# Train the model
model = lr.fit(train)

print("\n=== Model Training Completed ===")

# Predict customer churn
predictions = model.transform(test)

print("\n=== Customer Churn Predictions ===")
predictions.select(
    "features",
    "label",
    "prediction"
).show(truncate=False)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Stop Spark session
spark.stop()

Run the file 
python churn_prediction.py
