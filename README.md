# 🚀 Time Series Forecasting of Google Play Store Metrics using RNNs with Hyperparameter Tuning

This project presents a comprehensive workflow for time series forecasting on the Google Play Store dataset. It involves extensive data preprocessing, the implementation of multiple Recurrent Neural Network (RNN) architectures, and a systematic comparison of their performance after optimizing their hyperparameters using Keras Tuner.

## 🎯 Problem Statement and Goal of Project

The goal of this project was to forecast future trends of key metrics from the Google Play Store dataset, such as app ratings, reviews, and installs. Beyond simply making predictions, I aimed to systematically evaluate and compare the effectiveness of different RNN architectures—**SimpleRNN, LSTM, and GRU**—for this specific task. The core objective was to leverage automated hyperparameter tuning to find the most optimal configuration for each model, thereby demonstrating a robust and methodical approach to model development and selection.

-----

## 💡 Solution Approach

My approach was structured as a multi-stage process, moving from raw data to a final comparative analysis:

1.  **Data Cleaning and Preprocessing:** The raw dataset from `googleplaystore.csv` was rigorously cleaned. This involved handling missing values and converting columns with inconsistent formats (like 'Size', 'Installs', and 'Price') into clean, numerical types suitable for machine learning.
2.  **Feature Scaling and Sequencing:** Five key numerical features ('Rating', 'Reviews', 'Size', 'Installs', 'Price') were selected and scaled to a [0, 1] range using `MinMaxScaler`. The data was then transformed into time-series sequences, where a window of 5 consecutive data points was used to predict the subsequent 3 points.
3.  **Automated Hyperparameter Tuning:** I utilized the **Keras Tuner** library with the **Hyperband** algorithm to find the optimal architecture for four different models. The tuner systematically searched for the best combination of hyperparameters, such as the number of layers and the number of units in each layer, by optimizing for the lowest Validation Mean Absolute Error (`val_mae`).
4.  **Comparative Analysis:** The best-performing version of each of the four models (SimpleRNN, LSTM, GRU, and a combined LSTM+GRU) was retrieved. I then calculated their final Mean Absolute Error (MAE) on the validation set and generated a comparative plot to visually assess their predictive accuracy against the true values.

-----

## 🛠️ Technologies & Libraries

  * **Python**: The core programming language for the project.
  * **TensorFlow & Keras**: For building, training, and evaluating the deep learning models.
  * **Keras Tuner**: For performing efficient and automated hyperparameter optimization.
  * **Scikit-learn**: For data preprocessing, specifically using `MinMaxScaler`.
  * **Pandas & NumPy**: For data manipulation, cleaning, and numerical operations.
  * **Matplotlib & Seaborn**: For data visualization and plotting the final results.

-----

## 📊 Description about Dataset

The project uses the **"Google Play Store Apps"** dataset, which contains public data on applications from the store. The initial dataset required significant preprocessing to be usable for a time series model. After cleaning, the model was trained on five numerical features: **'Rating', 'Reviews', 'Size', 'Installs', and 'Price'**.

-----

## ⚙️ Installation & Execution Guide

To replicate this analysis, you can set up a Python environment and run the Jupyter Notebook.

1.  **Clone the repository and create a virtual environment:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow keras-tuner matplotlib seaborn
    ```
3.  **Execute the Notebook:**
    Launch Jupyter Notebook and open the `gru_2_completed_full_comparison.ipynb` file. Run the cells in sequential order to perform data preprocessing, model tuning, and evaluation.
    ```bash
    jupyter notebook
    ```

-----

## 📈 Key Results / Performance

After running the Keras Tuner to find the optimal hyperparameters for each architecture, the models were evaluated on the validation set. The Mean Absolute Error (MAE) for each model was as follows:

  * **SimpleRNN MAE:** 0.0435
  * **LSTM MAE:** 0.0401
  * **GRU MAE:** 0.0400
  * **Combined LSTM+GRU MAE:** 0.0403

The results indicate that the **GRU** model achieved a slightly better performance on this specific forecasting task compared to the other architectures.

-----

## 🖼️ Screenshots / Sample Output

The final output of the notebook is a comprehensive plot that visualizes the predictions of all four optimized models against the true validation data, providing a clear comparison of their performance.

-----

## 🧠 Additional Learnings / Reflections

This project was a valuable exercise in building a complete deep learning workflow. Key takeaways include:

  * **The Power of Automation:** Using Keras Tuner was significantly more efficient and robust than manual trial-and-error for finding optimal model hyperparameters. It allowed for a systematic and reproducible search process.
  * **Architecture Matters:** The direct comparison of SimpleRNN, LSTM, and GRU on the same dataset provided clear insights into their relative strengths. The more advanced architectures (LSTM and GRU) predictably outperformed the SimpleRNN.
  * **Data Preprocessing is Crucial:** The success of the models was heavily dependent on the initial data cleaning and feature engineering steps. Transforming inconsistent, real-world data into a clean, scaled format was a critical prerequisite for effective model training.

-----

## 👤 Author

## Mehran Asgari

## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)

## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

-----

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*