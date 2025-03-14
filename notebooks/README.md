# Jupyter Notebooks for EEG-based REM Sleep Detection

This directory contains Jupyter notebooks for exploring and visualizing EEG sleep data and its relationship with mood.

## Notebooks

### 1. Exploratory Data Analysis (`exploratory_data_analysis.ipynb`)

This notebook explores the relationship between EEG sleep patterns and mood upon waking. It includes:

- Analysis of sleep metrics distribution
- Visualization of sleep architecture
- Correlation analysis between sleep metrics and mood
- Dimensionality reduction for visualizing relationships
- Simple regression model for mood prediction

## Running the Notebooks

To run these notebooks, you need to have Jupyter installed. If you've installed the project dependencies, Jupyter should already be available.

From the project root directory, run:

```bash
jupyter notebook notebooks/
```

Or to run a specific notebook:

```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

## Adding New Notebooks

When adding new notebooks, please follow these guidelines:

1. Use clear, descriptive names for notebooks.
2. Include markdown cells with explanations of the analysis.
3. Add the notebook to this README with a brief description.
4. Keep code cells clean and well-commented.
5. Consider adding a summary or conclusions section at the end.

## Dependencies

The notebooks depend on the project's Python environment. Make sure you've installed all dependencies from the project's `requirements.txt` file:

```bash
pip install -r ../requirements.txt
``` 