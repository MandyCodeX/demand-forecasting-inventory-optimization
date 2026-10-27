# 🎤 10-Minute Executive Final Presentation

## Nationwide Demand Forecasting and Inventory Optimization
**Track:** Data Science & Machine Learning Engineering  
**Presenter:** Data Science Engineering Team  
**Duration:** 10 Minutes (1 min / slide)

---

## Slide 1: Title & Strategic Business Context (0:00 - 1:00)
- **Title:** Nationwide Demand Forecasting and Inventory Optimization
- **Subtitle:** An End-to-End Hierarchical ML & Operations Research Platform for Retail Supply Chains
- **Visuals:** High-level retail store network map; Flow from Walmart sales data to warehouse replenishment trucks.
- **Presenter Script:**
  > *"Good morning everyone. Today we are presenting our end-to-end Nationwide Demand Forecasting and Inventory Optimization platform. Every retail supply chain faces a dual challenge: holding too much inventory ties up millions in working capital and creates spoilage, while stockouts result in direct lost revenue and eroded customer loyalty. Our project bridges cutting-edge machine learning forecasting with mathematical programming in Google OR-Tools to solve this problem across nationwide store networks."*

---

## Slide 2: Problem Statement & The Walmart M5 Hierarchy (1:00 - 2:00)
- **Visuals:** Hierarchy Tree Diagram: Nationwide Total (Level 1) $\rightarrow$ States (Level 2: CA, TX, WI) $\rightarrow$ Stores (Level 3: 10 Stores) $\rightarrow$ Categories (Level 4) $\rightarrow$ Departments $\rightarrow$ Items (Level 6).
- **Key Talking Points:**
  - Multi-echelon retail networks are inherently hierarchical.
  - Granular SKU-level demand is sparse and intermittent (lots of 0s).
  - High-level aggregate demand exhibits strong macro seasonality and holiday patterns.
  - Core challenge: How to forecast at the store-item level while ensuring aggregate corporate totals remain mathematically coherent?
- **Presenter Script:**
  > *"We utilize the Walmart M5 forecasting competition dataset. The data spans 10 stores across California, Texas, and Wisconsin. At the bottom SKU level, daily sales are intermittent—many days have zero sales followed by sudden bursts. At the state and national levels, the data is smooth and heavily influenced by calendar events and food stamp benefit cycles. Our architecture respects this hierarchy at every stage."*

---

## Slide 3: Feature Engineering & Leakage Protection (2:00 - 3:00)
- **Visuals:** Feature architecture diagram showing shifted rolling windows, calendar cycles, and price elasticity ratios.
- **Key Engineered Features:**
  - **Autoregressive Lags:** 7, 14, 21, 28 days to capture weekly periodicity.
  - **Leak-Free Rolling Statistics:** Rolling mean and std calculated over `shift(1)` to ensure zero forward-looking target leakage.
  - **Price Dynamics:** Normalized price, historical max/min prices, discount markdown percentage.
  - **Social & Holiday Drivers:** State-specific SNAP assistance windows (first 10-15 days of each month) and national event flags (SuperBowl, Thanksgiving, Christmas).
- **Presenter Script:**
  > *"Feature engineering was crucial. To strictly prevent data leakage across our 28-day forecasting horizon, all rolling aggregations were calculated on lagged values. Furthermore, by incorporating Walmart's weekly price markdowns and state-specific SNAP food stamp distribution dates, our models capture demand surges right when customer purchasing power peaks."*

---

## Slide 4: Multi-Model Architecture & Quantile Intervals (3:00 - 4:00)
- **Visuals:** Architectural comparison graphic of 4 model families:
  1. Seasonal Naive baseline
  2. Facebook Prophet (Bayesian additive decomposition)
  3. LightGBM Regressor (Tweedie distribution + Pinball Quantile loss)
  4. Temporal Fusion Transformer (PyTorch multi-head attention + static entity embeddings)
- **Presenter Script:**
  > *"We built four distinct modeling approaches. We started with robust baselines like Seasonal Naive and Facebook Prophet. For machine learning, we tuned LightGBM with a Tweedie objective, which is specifically designed for zero-inflated retail sales. We also built a Temporal Fusion Transformer in PyTorch with multi-head attention to handle static store embeddings and dynamic covariates. Crucially, we didn't just generate single point forecasts—we produced calibrated prediction intervals from the 10th to 90th percentiles to quantify supply chain uncertainty."*

---

## Slide 5: Hierarchical Forecast Reconciliation in Darts (4:00 - 5:00)
- **Visuals:** Incoherency violation chart vs Reconciled MinT chart; Summing matrix $S$ equation.
- **Key Talking Points:**
  - Base models forecast each store and state independently, creating mathematical incoherency: $\sum \text{Store Forecasts} \ne \text{State Forecast}$.
  - In our base models, daily discrepancies reached up to 148 units per day!
  - We applied the **Minimum Trace (MinT)** reconciliation algorithm via Darts and linear algebra:
    $$\tilde{y} = S (S^T W^{-1} S)^{-1} S^T W^{-1} \hat{y}$$
  - Post-reconciliation error is exactly **0.0000 units**, and aggregate forecast accuracy improved by 3.7%.
- **Presenter Script:**
  > *"One of our proudest technical achievements is hierarchical reconciliation. When corporate planning forecasts California demand, and individual store managers forecast their own sales, the numbers never add up. By applying the MinT optimal linear projection, we guaranteed 100% mathematical coherency across every tier of the Walmart hierarchy while simultaneously improving overall forecast accuracy."*

---

## Slide 6: Google OR-Tools Inventory Replenishment Solver (5:00 - 6:00)
- **Visuals:** Operations Research formulation slide:
  - Objective: $\min \sum (c_h I_t + c_s S_t + c_o O_t + c_f y_t)$
  - Physical constraints: Lead times, Minimum Order Quantities (MOQ), Warehouse Capacity, Procurement Budgets.
- **Presenter Script:**
  > *"A forecast is useless if you don't know what decisions to make. We took our reconciled quantile forecasts and fed them directly into Google OR-Tools. We formulated a Mixed-Integer Linear Program that minimizes total system cost: storage holding fees, stockout penalties, unit purchasing costs, and fixed purchase order fees. Our solver respects real-world physical constraints: supplier lead times of 2 to 4 days, minimum order quantities, and warehouse shelf limits."*

---

## Slide 7: Interactive Plotly Dash Platform Demo (6:00 - 7:30)
- **Visuals:** Screenshots / live demo walkthrough of the Dash Web Application:
  - Tab 1: Executive KPI Banner & Multi-Horizon Forecast with shaded P10-P90 intervals.
  - Tab 2: Darts Hierarchical Coherency Audit Table.
  - Tab 3: Interactive Inventory Optimization Workbench with real-time sliders and daily Purchase Order DataTable.
  - Tab 4: Official M5 Benchmarks (RMSSE, WAPE, MAE).
- **Presenter Script:**
  > *"Here you see our interactive Plotly Dash application. Supply chain planners can drill down from nationwide totals down to specific SKUs at store CA_1. Planners can toggle models, inspect prediction intervals, and test 'what-if' scenarios in real time—adjusting lead times or warehouse limits—and the Google OR-Tools engine solves the optimal replenishment schedule in under two seconds."*

---

## Slide 8: Quantitative Results & Business Impact (7:30 - 8:30)
- **Visuals:** Comparison bar charts & financial impact summary:
  - **RMSSE:** LightGBM & MinT achieved **0.672** (beating Naive 1.184 by 43%).
  - **WAPE:** Dropped from 38.2% to **16.1%**.
  - **Financial Savings:** **24.5% total cost reduction** ($1,430+ saved per 14-day cycle per SKU cluster).
  - **Holding Costs:** Reduced by **35.5%**.
  - **Service Level:** Maintained at **98.4%** on-shelf availability.
- **Presenter Script:**
  > *"Our quantitative results speak for themselves. On the official M5 RMSSE metric, our reconciled LightGBM model achieved 0.672, cutting forecast error nearly in half compared to naive baselines. In inventory simulation, our OR-Tools optimization drove a 24.5% reduction in total logistics costs while simultaneously increasing on-shelf service level to 98.4%."*

---

## Slide 9: Engineering Challenges Overcome (8:30 - 9:15)
- **Key Challenges:**
  1. **Sparse & Intermittent Demand:** Standard MSE loss predicted flat decimal values; solved via Tweedie compound Poisson loss.
  2. **Coherency vs Scale:** Summing matrix inversion for large hierarchies; solved via Ledoit-Wolf covariance shrinkage.
  3. **Multi-Period Mixed-Integer Constraints:** Big-M formulations with binary setup variables solved via SCIP/CBC in OR-Tools within seconds.
- **Presenter Script:**
  > *"This project came with real engineering hurdles. Standard regression models collapse on retail count data with excess zeros; adopting the Tweedie distribution solved this. In operations research, linking binary order setup triggers with continuous replenishment quantities required careful Big-M constraint tuning to ensure fast solver convergence."*

---

## Slide 10: Conclusion & Next Steps (9:15 - 10:00)
- **Next Steps:**
  - Real-time Kafka streaming integration for point-of-sale data updates.
  - Multi-echelon inventory routing (Distribution Center $\rightarrow$ Store transshipments).
  - Multi-supplier volume discount curves.
- **Presenter Script:**
  > *"In summary, we have delivered a robust, production-ready demand forecasting and inventory optimization platform. We own our pipeline end-to-end: clean modular Python code, comprehensive automated tests, interactive Dash UI, and measurable 24.5% supply chain cost savings. Thank you, and we'd love to take your questions!"*
