# Amazon Co-Purchase Link Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research implementation of **time-aware link prediction** on Amazon co-purchase networks with **leakage-free temporal evaluation**. This project demonstrates rigorous evaluation practices for link prediction on temporal graphs, avoiding common pitfalls like training on future information.

## Project Overview

This project builds co-purchase networks from Amazon review data where nodes are products and edges connect products frequently reviewed together. We predict future co-purchases using only historical information through a temporal train/test split at the **80th percentile timestamp** (t\*), ensuring no information leakage from the future.

The evaluation covers three Amazon product categories (Electronics, All_Beauty, Home_and_Kitchen) and compares classical link prediction baselines under both **warm-start** (seen nodes) and **cold-start** (unseen nodes) scenarios.

**Authors:** Yeseul Han, Erfan YousefMoumji, Xi Lu

## Key Contributions

- **Leakage-free temporal split**: Train on edges up to t\* (80th percentile), test on edges after t\*
- **2-hop candidate generation**: Efficient negative sampling from nodes at distance 2
- **Warm vs cold-start evaluation**: Separate analysis for seen vs unseen test nodes
- **Baseline implementations**: Common Neighbors (CN), Jaccard, Adamic-Adar (AA), Preferential Attachment (PA)
- **Comprehensive metrics**: AUC, Average Precision (AP), Precision@K (K=10,20,50,100,200,500)
- **Multi-seed evaluation**: 5 random seeds per configuration for statistical robustness
- **Runtime profiling**: Performance analysis across categories and baselines

## Repository Structure

```
.
├── README.md                      # This file
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── CITATION.cff                   # Citation metadata
├── Makefile                       # Build automation (optional)
├── .gitignore                     # Git ignore rules
│
├── src/
│   ├── projectmilestone2.py      # Main pipeline implementation
│   └── run.py                     # Simple entry point 
│
├── data/
│   └── README.md                  # Dataset download instructions
│   # Dataset files go here (not committed)
│
├── results/
│   # CSV files with evaluation metrics
│   ├── 5core_stats.csv
│   ├── eval_linkpred_all.csv
│   ├── eval_linkpred_cold.csv
│   ├── overall_metrics.csv
│   └── coldstart_metrics.csv
│
├── figs/
│   # Key plots and visualizations
│   ├── graph_stats_nodes_edges.png
│   ├── graph_stats_density.png
│   ├── edge_time_cdf_*.png
│   └── ...
│
└── report/
    └── Final_report.pdf           
```

## How to Run

### Prerequisites

1. **Python 3.7+** with pip installed
2. **Download datasets**: Follow instructions in [`data/README.md`](data/README.md) to download the Amazon 5-core datasets

### Installation

```bash
# Clone the repository
git clone https://github.com/ErfanYM/amazon-copurchase-link-prediction.git
cd amazon-copurchase-link-prediction

# Install dependencies
pip install -r requirements.txt
```

### Download Data

See [`data/README.md`](data/README.md) for detailed instructions. You need:
- `Electronics_5.json.gz`
- `All_Beauty_5.json.gz`
- `Home_and_Kitchen_5.json.gz`

Place them in the `data/` directory.

### Run the Pipeline

**Option 1: Using Make (recommended)**

```bash
# Install dependencies
make install

# Run the pipeline
make run

# Clean results (if needed)
make clean
```

**Option 2: Using the run script directly**

```bash
python src/run.py --data-dir data
```

**Option 3: Direct execution (after updating paths in code)**

```bash
# Edit src/projectmilestone2.py to update data paths (lines 53-55)
# Then run:
python src/projectmilestone2.py
```

The pipeline will:
1. Load and sample 100K reviews per category
2. Build co-purchase graphs and extract giant connected components (GCC)
3. Compute edge timestamps and determine t\* (80th percentile)
4. Split edges into train/test sets (warm and cold-start)
5. Generate 2-hop candidates for link prediction
6. Evaluate 4 baselines × 3 categories × 5 seeds
7. Save results to `results/` and plots to `figs/`

**Expected runtime**: ~10-30 minutes depending on hardware

## Outputs

### Results CSVs

- `results/5core_stats.csv`: Graph statistics (nodes, edges, density, avg degree)
- `results/eval_linkpred_all.csv`: Per-seed warm evaluation results
- `results/eval_linkpred_cold.csv`: Per-seed cold-start evaluation results
- `results/overall_metrics.csv`: Aggregated warm metrics (mean ± std)
- `results/coldstart_metrics.csv`: Aggregated warm + cold metrics

### Figures

- `figs/graph_stats_nodes_edges.png`: Graph size comparison
- `figs/graph_stats_density.png`: Density comparison
- `figs/edge_time_cdf_*.png`: Edge timestamp CDFs with t\* cutoff
- Additional plots for AUC/AP/Precision@K comparisons

## Dataset

This project uses the **Amazon 5-core review datasets** (McAuley et al., 2019). The raw data is **NOT included** in this repository.

- **Download**: [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html)
- **Citation**: See [`data/README.md`](data/README.md) for proper attribution

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@software{han_yousefmoumji_lu_2026_amazon_linkpred,
  author = {Han, Yeseul and YousefMoumji, Erfan and Lu, Xi},
  title = {Amazon Co-Purchase Link Prediction: Time-Aware, Leakage-Free Evaluation},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/ErfanYM/amazon-copurchase-link-prediction}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Amazon review datasets by Julian McAuley et al.
- NetworkX library for graph analysis
- scikit-learn for evaluation metrics

---

**Maintainer**: [Erfan YousefMoumji](https://github.com/ErfanYM)  
**Last Updated**: January 2026
