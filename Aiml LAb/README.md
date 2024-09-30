
# Exploratory Data Analysis (EDA) of Supplier and Parts Dataset

## Overview

This project involves performing **Exploratory Data Analysis (EDA)** on a dataset that contains information about suppliers, parts, their quantities, prices, and locations. The goal of this project is to provide insights into the dataset by summarizing key statistics, identifying patterns, detecting missing values, and visualizing relationships within the data.

## Dataset Description

The dataset contains information about suppliers, parts, and associated metrics. It is structured into 5 columns with each row representing a unique combination of supplier and part information. Below is a description of the columns in the dataset:

- **Supplier**: The name of the supplier (e.g., Supplier A, Supplier B, etc.).
- **Part**: The name or type of the part being supplied (e.g., Part X, Part Y, etc.).
- **Quantity**: The number of parts supplied.
- **Price**: The price of the part in monetary units.
- **Location**: The location (e.g., city or state) where the supplier operates.

### Sample Data:

| Supplier   | Part   | Quantity | Price | Location |
|------------|--------|----------|-------|----------|
| Supplier A | Part X | 100      | 20.5  | NY       |
| Supplier B | Part Y | 150      | 30.0  | LA       |
| Supplier C | Part Z | 200      | 25.5  | SF       |
| Supplier D | Part W | 50       | 22.0  | NY       |
| Supplier E | Part V | 300      | 35.0  | TX       |

## Analysis Process

The following steps were taken to perform the exploratory data analysis:

1. **Loading the Dataset**: The dataset is imported into the Python environment using the `pandas` library.
2. **General Information**: The structure and types of the dataset are displayed using `info()`, showing the number of non-null values, data types, and memory usage.
3. **Statistical Summary**: Basic descriptive statistics (mean, median, standard deviation, etc.) are generated using `describe()` for both numerical and categorical columns.
4. **Missing Values**: Missing values are checked, and a heatmap is generated to visually inspect the distribution of missing data. In this case, no missing values were found.
5. **Correlation Matrix**: A correlation matrix is generated to show the relationship between numerical columns like `Quantity` and `Price`. This is visualized with a heatmap to better understand how features correlate with one another.
6. **Distribution of Numerical Data**: Histograms are plotted for numerical columns to show the distribution of values.
7. **Outlier Detection**: Boxplots are created for numerical columns to detect potential outliers in the dataset.
8. **Categorical Analysis**: Unique values and their frequency are calculated for each categorical column. Count plots are also created to visualize the distribution of categorical data.

## Visualizations

Several visualizations were created to better understand the dataset:

1. **Heatmap of Missing Values**: Shows if any missing values exist in the dataset.
2. **Correlation Heatmap**: Visualizes the relationships between numerical features.
3. **Histograms**: Display the distribution of numerical data.
4. **Boxplots**: Help detect outliers in numerical features.
5. **Count Plots**: Show the distribution of categorical variables like `Supplier`, `Part`, and `Location`.

## Dependencies

To run the analysis, the following Python libraries are required:

- **pandas**: For data manipulation and analysis.
- **matplotlib**: For plotting graphs.
- **seaborn**: For creating beautiful visualizations like heatmaps and boxplots.

You can install these libraries using `pip`:

```bash
pip install pandas matplotlib seaborn
```

## How to Run the Project

1. Clone or download this repository.
2. Make sure you have the `data.csv` file in the same directory as the Python script.
3. Run the Python script using your preferred IDE or command line:
   ```bash
   python analysis_script.py
   ```
4. Visualize the outputs, including the plots and statistical summaries.

## Conclusion

This project provides an introductory analysis of a simple supplier-parts dataset. It gives a solid overview of the structure, key statistics, and potential relationships within the data. Future steps could include deeper analysis or applying machine learning techniques to derive predictive insights from the data.

## License

This project is open-source and free to use under the MIT License.
