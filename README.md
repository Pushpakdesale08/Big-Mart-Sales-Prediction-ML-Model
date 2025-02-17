# Big-Mart-Sales-Prediction-ML-Model

## Project Overview
This project aims to predict sales for Big Mart stores based on various features such as store type, location, item details, and historical sales data. The model will help retailers understand key factors affecting sales and optimize inventory management.

## Dataset
The dataset consists of historical sales data from multiple Big Mart stores, including:
- `Item_Identifier`: Unique identifier for items
- `Item_Weight`: Weight of the item
- `Item_Fat_Content`: Fat content of the item
- `Item_Visibility`: The percentage visibility of the item in the store
- `Item_Type`: Category of the item
- `Item_MRP`: Maximum Retail Price of the item
- `Outlet_Identifier`: Store ID
- `Outlet_Establishment_Year`: Year when the store was established
- `Outlet_Size`: Store size (Small/Medium/Large)
- `Outlet_Location_Type`: Type of city (Tier 1, Tier 2, etc.)
- `Outlet_Type`: Type of outlet (Supermarket, Grocery Store, etc.)
- `Item_Outlet_Sales`: Target variable - Sales of the item in the given store

## Objective
The goal is to build a machine learning model that accurately predicts `Item_Outlet_Sales` based on the available features.

## Steps Involved
1. **Exploratory Data Analysis (EDA)**
   - Understanding data distribution
   - Identifying missing values and handling them
   - Outlier detection and treatment
   - Feature engineering

2. **Data Preprocessing**
   - Handling categorical variables (Label Encoding)
   - Feature scaling
   - Imputing missing values

5. **Model Selection & Training**
   - Linear Regression
   - Random Forest
   - XGBoost

6. **Model Evaluation**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R-squared Score (R²)

## Results & Insights
- Analyzed the importance of various features affecting sales
- Optimized model performance with hyperparameter tuning
- Provided insights into how different store types influence sales

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, XGBoost)
- **Jupyter Notebook** for development
- **Git/GitHub** for version control

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Big-Mart-Sales-Prediction.git
   cd Big-Mart-Sales-Prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `Big_Mart_Sales_Prediction.ipynb` and execute the cells.

## Future Enhancements
- Incorporating external economic data for better predictions
- Using deep learning models for improved accuracy
- Deployment of the model as a web application

## Contributing
Feel free to fork this repository and submit pull requests if you'd like to contribute!

## License
This project is open-source and available under the MIT License.

