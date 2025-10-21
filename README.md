<!-- # Granger-Causality

Python implementation of the **Multivariate Granger Causality (MVGC) Toolbox**.

This project provides a Python translation of the original MATLAB toolbox introduced in the following paper:

> **The MVGC Multivariate Granger Causality Toolbox: A New Approach to Granger-Causal Inference**  
> *Lionel Barnett and Anil K. Seth*  
> *Journal of Neuroscience Methods, Vol. 223, 2014, pp. 50–68*  
> [https://doi.org/10.1016/j.jneumeth.2013.10.018](https://doi.org/10.1016/j.jneumeth.2013.10.018)

### Reference
If you use this implementation in your research, please cite:
```bibtex
@article{barnett2014mvgc,
  title   = {The MVGC multivariate Granger causality toolbox: a new approach to Granger-causal inference},
  author  = {Barnett, Lionel and Seth, Anil K},
  journal = {Journal of Neuroscience Methods},
  volume  = {223},
  pages   = {50--68},
  year    = {2014},
  publisher = {Elsevier}
}
```

### Notes
- The original MATLAB version can be found on [Lionel Barnett’s website](https://users.sussex.ac.uk/~lionelb/).
- This Python version aims to preserve the core logic and functionality while improving readability and integration with modern scientific Python libraries (`numpy`, `scipy`, `statsmodels`, etc.). -->



# 🧠 Multivariate Granger Causality (MVGC) — Python Implementation

A **Python translation** of the **Multivariate Granger Causality (MVGC) Toolbox**, originally developed in MATLAB by *Lionel Barnett* and *Anil K. Seth*.

This project provides a modular, numerically stable, and readable Python port of the core MVGC algorithms, preserving their mathematical foundations while integrating with modern Python scientific libraries (`NumPy`, `SciPy`, `Statsmodels`, etc.).

---

## 📘 Reference Paper

> **The MVGC Multivariate Granger Causality Toolbox: A New Approach to Granger-Causal Inference**
> *Lionel Barnett and Anil K. Seth*
> *Journal of Neuroscience Methods, Vol. 223, 2014, pp. 50–68*
> [https://doi.org/10.1016/j.jneumeth.2013.10.018](https://doi.org/10.1016/j.jneumeth.2013.10.018)

If you use this implementation in your research, please cite:

```bibtex
@article{barnett2014mvgc,
  title={The MVGC multivariate Granger causality toolbox: a new approach to Granger-causal inference},
  author={Barnett, Lionel and Seth, Anil K},
  journal={Journal of Neuroscience Methods},
  volume={223},
  pages={50--68},
  year={2014},
  publisher={Elsevier}
}
```

---

## 🧩 Project Structure

Below is an overview of each Python module and its purpose within the pipeline.

### `tsdata_to_var.py`

* **Purpose:** Estimates a Vector Autoregressive (VAR) model from time-series data.
* **Function:** `tsdata_to_var(X, p)`
* **Description:**
  Implements the LWR (Levinson–Wiggins–Robinson) recursion for multivariate time series to compute the VAR coefficients (`A`), residual covariance (`SIG`), and residuals (`E`).
  Equivalent to MATLAB’s `tsdata_to_var` from MVGC.

---

### `tsdata_to_infocrit.py`

* **Purpose:** Determines optimal model order using information criteria.
* **Functions:**

  * `tsdata_to_infocrit(X, morder)`: Computes AIC and BIC across possible model orders.
  * `infocrit(L, k, m)`: Computes corrected Akaike (AIC) and Bayesian (BIC) information criteria.
  * `_demean(X, normalize)`: Demeans or normalizes the data for preprocessing.
* **Description:**
  Provides a self-contained method for model order selection using forward–backward residual estimation and Cholesky-based updates.

---

### `var_to_autocov.py`

* **Purpose:** Computes the autocovariance sequence from a fitted VAR model.
* **Function:** `var_to_autocov(A, SIG)`
* **Description:**
  Solves a discrete Lyapunov equation via the Schur decomposition (`lyapslv.py`) to obtain the theoretical autocovariance sequence of a VAR process. Also computes key stability diagnostics (e.g., spectral radius `ρ`).

---

### `autocov_to_var.py`

* **Purpose:** Estimates VAR parameters directly from the autocovariance sequence.
* **Function:** `autocov_to_var(G)`
* **Description:**
  Implements the **Whittle recursion** to reconstruct forward and backward prediction coefficients (`AF`, `AB`) and estimate noise covariance (`SIG`).
  Used as the inverse operation of `var_to_autocov`.

---

### `autocov_to_var_optimized.py`

* **Purpose:** Optimized and numerically stable version of `autocov_to_var.py`.
* **Function:** `autocov_to_var_optimized(G)`
* **Description:**
  Rewrites critical matrix operations using `np.linalg.solve` for improved speed and numerical stability on large datasets.
  Recommended for GPU or high-dimensional use cases.

---

### `autocov_to_pwcgc.py`

* **Purpose:** Computes pairwise-conditional Granger causality (PWCGC) from autocovariances.
* **Function:** `autocov_to_pwcgc(G, SIG)`
* **Description:**
  Iterates through variable pairs and computes Granger causality values based on log ratio of residual variances.
  Follows the definition ( F_{i \to j} = \ln(\Sigma_j / \Sigma_{j|i}) ).

---

### `lyapslv.py`

* **Purpose:** Solves continuous/discrete Lyapunov equations using the Schur method.
* **Function:** `lyapslv(A, Q)`
* **Description:**
  Performs a **Schur decomposition-based** solution for ( A X A^T - X + Q = 0 ).
  This function ensures stable computation of covariance sequences for VAR systems, matching MATLAB’s `lyapslv` implementation.

---

## ⚙️ Dependencies

* Python ≥ 3.9
* NumPy ≥ 1.23
* SciPy ≥ 1.9
* (Optional) Statsmodels ≥ 0.14 — for validation and comparison with built-in VAR tools

Install all dependencies:

```bash
pip install numpy scipy statsmodels
```

---

## 🧪 Example Usage

```python
import numpy as np
from tsdata_to_var import tsdata_to_var
from var_to_autocov import var_to_autocov
from autocov_to_pwcgc import autocov_to_pwcgc

# Generate dummy data
np.random.seed(0)
X = np.random.randn(3, 200, 1)  # 3 variables, 200 samples, 1 trial

# Estimate VAR(2)
A, SIG, E = tsdata_to_var(X, p=2)

# Convert to autocovariance sequence
G = var_to_autocov(A, SIG)

# Compute pairwise Granger causality
F = autocov_to_pwcgc(G, SIG)
print("Pairwise GC matrix:\n", F)
```

---

## 📁 Directory Overview

```
.
├── tsdata_to_var.py             # Estimate VAR model from time series
├── tsdata_to_infocrit.py        # Compute AIC/BIC to select optimal order
├── var_to_autocov.py            # Generate autocovariance sequence from VAR
├── autocov_to_var.py            # Reconstruct VAR from autocovariance
├── autocov_to_var_optimized.py  # Optimized version for speed/stability
├── autocov_to_pwcgc.py          # Compute pairwise-conditional GC
├── lyapslv.py                   # Solve Lyapunov equation using Schur
└── README.md                    # Project documentation
```

---

## 📜 License

This project is provided for research and educational purposes only.
Please acknowledge the original authors when using or adapting the code.

---

✫️ *Maintained by Mohammad Hossein Moslemi*
Western University — Department of Computer Science
