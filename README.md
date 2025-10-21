# Granger-Causality

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
- This Python version aims to preserve the core logic and functionality while improving readability and integration with modern scientific Python libraries (`numpy`, `scipy`, `statsmodels`, etc.).
