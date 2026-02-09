# Turkish Super League Attendance Rate Predictor ⚽

This repository contains a **Hybrid Artificial Intelligence System** designed to predict stadium occupancy rates for football matches in the **Turkish Super League (Süper Lig)**.

Unlike traditional regression models, this project combines **Gradient Boosting Machine Learning** with **Rule-Based Statistical Inference** to account for the unique sociological dynamics of Turkish football, such as the "Big 4" dominance, derby effects, and the asymmetry between home and away team support.

## Key Features
* **Hybrid Architecture:** Uses a weighted ensemble of **Rule-Based Statistical Inference (65%)** and **Gradient Boosting Regressor (35%)** to minimize overfitting on a limited dataset.
* **Context-Aware Ranking:** Implements a custom **1-to-5 Rank System** (Championship Tier, European Spot, Mid-Table, Relegation) to handle league size fluctuations caused by the 2023 Earthquake.
* **Derby Detection:** Specifically models the attendance spike (approx. 3x) observed in matches between Galatasaray, Fenerbahçe, Beşiktaş, and Trabzonspor.
* **Fallback Mechanism:** Includes a hierarchical fallback system that prioritizes historical data matches before relying on generalized ML predictions.

## Project Structure

| File | Description |
| :--- | :--- |
| `final_gradient_boosting.py` | The main source code. Contains the hybrid model logic, data preprocessing, and the interactive prediction interface. |
| `superlig_attendance.xlsx` | The comprehensive dataset containing match statistics, attendance rates, and rankings for the 2022-2025 seasons. |
| `ANN FINAL PRESENTATION.pptx` | The official project presentation slides covering the methodology, analysis, and results. |
| `youtube.url` | A direct link to the project demonstration video. |

## Dataset & Methodology

### 1. Data Collection
The dataset (`superlig_attendance.xlsx`) consists of **506 matches** from the last 3 seasons (2022-2025), collected from *Transfermarkt* and *TFF*.
* **Inputs:** Home Team, Away Team, Home Rank (1-5), Away Rank (1-5), Derby Status, Stadium Capacity.
* **Target:** Occupancy Rate (Attendance / Capacity).

### 2. The Hybrid Approach
The model addresses the non-linear behavior of Turkish fans using two pipelines implemented in `final_gradient_boosting.py`:
1.  **Statistical Inference:** Analyzes historical matchups and specific team behaviors.
2.  **Machine Learning (Gradient Boosting):** Captures complex patterns and categorizes team interactions.
3.  **Final Prediction:**
    $$P_{final} = 0.65 \times P_{Statistical} + 0.35 \times P_{ML}$$

## Performance Insights
The model was evaluated using "Actual vs. Predicted" analysis, revealing key insights into Turkish football culture:

* **The "Big 4" Effect:** For major teams, attendance is driven by their own league rank.
* **The Anatolian Effect:** For smaller teams, attendance is heavily influenced by the **Away Team's rank** (fans attending to see the "Big 4" opponents).
* **Derby Impact:** Derby matches show consistently higher occupancy rates (near 90-100%) compared to standard matches (~30-40%).

## Installation & Usage

### Prerequisites
* Python 3.8+
* Required libraries: `pandas`, `scikit-learn`, `matplotlib`, `numpy`, `openpyxl`

### Running the Predictor
1.  Clone the repository:
    ```bash
    git clone [https://github.com/alkana25/Turkish-Super-League-Attendance-Rate.git](https://github.com/alkana25/Turkish-Super-League-Attendance-Rate.git)
    cd Turkish-Super-League-Attendance-Rate
    ```

2.  Run the main script:
    ```bash
    python final_gradient_boosting.py
    ```

3.  **Interactive Mode:** The script will prompt you to enter teams and their current ranks (1-5) to generate a prediction.

    ```text
    Home team: Galatasaray
    Away team: Fenerbahce
    Home rank (1-5): 1
    Away rank (1-5): 1
    --- RESULTS ---
    Estimated Attendance: 51,906
    Occupancy Rate: 96.2%
    ```

---
*This project was developed as the Final Project for the **EHB 420E - Artificial Neural Networks** course at Istanbul Technical University.*
