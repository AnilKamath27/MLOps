<h1 align="center">🚗 Used Car Selling Price Prediction</h1>

<p align="center">
  <img src="_assets\1_ZOcUPrSXLYucFxppoI-dYg.png" alt="Car Selling Price Prediction" width="1400" height="400">
</p>

In this Mlops version of Porject Used Car Selling Price Prediction, I've implemented an MLOps version of model building, following a structured and industry-standard approach known as "Modular Coding." This approach is widely practiced in the field, ensuring the development of robust and maintainable machine learning projects. I've integrated essential tools like ZenML and leveraged the power of MLflow to enhance the overall project management and monitoring.

Here's how I've approached this advanced coding format:

1. Structured Workflow: I've organized the project into a structured workflow, separating different phases of the machine learning pipeline, such as data preprocessing, model training, and evaluation. This makes it easier to maintain and scale the project.

2. Resource Attribution: I want to acknowledge the valuable resources that I've relied on to implement this MLOps version effectively. These include:
- [Build ML Production Grade Projects For Free | MLOps Course For Beginners](https://youtu.be/dPmH3G9NQtY?si=6RtwvqZ-RE6aMJFr): This course has provided insights into best practices for developing production-grade ML projects.
- [Zenml official Github](https://github.com/zenml-io/zenml): ZenML is an essential tool for versioning and managing machine learning pipelines, making the project more reproducible and collaborative.
- [Mlflow official page](https://mlflow.org/docs/latest/introduction/index.html): MLflow has been integrated to streamline experiment tracking, model management, and deployment.

3. Industry Best Practices: I've followed industry best practices for developing machine learning projects, such as modular code design, version control, and model tracking. This ensures that the project is maintainable, and it's easier to collaborate with team members.

4. MLOps Integration: By integrating ZenML and MLflow, I've added a layer of MLOps (Machine Learning Operations) to the project. This allows for end-to-end management of the machine learning pipeline, from data ingestion to model deployment, making it more efficient and robust.

Overall, this project is a demonstration of a high-quality, MLOps-compliant machine learning project. It follows industry standards, leverages powerful tools, and incorporates best practices, setting a strong foundation for building and deploying machine learning models effectively.
## Table of Contents

- [Scope of the Project](#Scope-of-the-Project)
- [Getting Started](#getting-started)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Selection](#model-selection)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)

---
## 🌐 Scope of the Project

The Car Selling Price Prediction project aims to develop a machine learning model that predicts the selling price of used cars based on various key features. The project includes data analysis, data preprocessing, model building, and hyperparameter tuning using the XGBoost algorithm.

## Getting Started

To get started with this project, you will need to have Python and the required libraries installed. You can install the necessary packages using pip:
```
pip install -r requirements.txt
```

Clone this repository to your local machine:
```
git clone https://github.com/AnilKamath27/MLOps.git
```
---
## 🔍 Exploratory Data Analysis
The EDA phase includes data visualization to understand the relationships between different features and the target variable (Selling Price). Some key EDA findings are:

- 🔍 Scatter plots to visualize the relationship between the year and selling price.
- 🔍 Box plots to identify outliers in key features.
- 🔍 Violin plots to visualize the distribution of selling prices for different transmission types.
- 🔍 Count plots for car names and fuel types.

## 🛠️ Data Preprocessing
In the data preprocessing phase:

- 🛠️ Duplicate records in the dataset were removed.
- 🛠️ Categorical variables were encoded using Label Encoding.
- 🛠️ Data was split into training and testing sets.
- 🛠️ Features were standardized using StandardScaler.

## 🤖 Model Selection
Four different machine learning models were considered for this project:

- 🤖 Linear Regression
- 🤖 Random Forest Regressor
- 🤖 Gradient Boosting Regressor
- 🤖 XGBoost Regressor

Our evaluation process included:

- Cross-validation to assess model performance using metrics like R-squared (R²), Mean Squared Error (MSE), and Mean Absolute Error (MAE).

## 🔧 Hyperparameter Tuning
We fine-tuned the XGBoost Regressor using Optuna, an automatic hyperparameter optimization framework. By optimizing the model's hyperparameters, we improved its performance, resulting in reduced MAE and MSE.

## 📊 Results
The XGBoost Regressor with the best hyperparameters was selected as the final model. Here are some key results:

- 📊 R-squared (R2): 0.9
- 📊 Mean Absolute Error (MAE): 0.87
- 📊 Mean Squared Error (MSE): 2.55

The model can be used to predict the selling price of used cars.

## 📈 Conclusion
In conclusion, this project provides insights into building a used car selling price prediction model. By exploring the dataset, preprocessing the data, and applying various machine learning models, we gained a deeper understanding of the factors that influence car prices. The project emphasizes the importance of data analysis, model training, and hyperparameter tuning.

Whether you're a seller looking to determine the fair market value of your car or a buyer seeking to assess the reasonability of a listed price, this project equips you with the knowledge to make informed decisions in the used car market.

Feel free to use this project as a reference or template for your own car price prediction projects. Remember, with a larger dataset, advanced feature engineering, and more extensive hyperparameter tuning, you can further enhance model performance.

Thank you for joining us on this journey through the world of used car selling price prediction! 🚗📊🛠️🔧📈

## How to Use
To use the trained model for car selling price prediction, you can follow these steps:

- Install the required packages (see "Getting Started").
- Clone this repository.
- Open and run the provided Jupyter Notebook or Python script.
- Input the car's features, and the model will predict the selling price.

## Contributing
If you'd like to contribute to this project, feel free to create a pull request or open an issue.

## Other reources used in this project

- [Why Are Manual Transmissions Still Popular in India?](https://www.spinny.com/blog/index.php/why-are-manual-transmissions-still-popular-in-india/)
- [India Car Sales by Transmission Type](https://www.statista.com/statistics/1037820/india-car-sales-by-transmission-type/) 
- [More than a third of all used cars sold were those with auto transmissions](http://timesofindia.indiatimes.com/articleshow/81263349.cms?utm_source=contentofinterest&utm_medium=text&utm_campaign=cppst)

## License
This project is licensed under the Apache License 2.0 License.
