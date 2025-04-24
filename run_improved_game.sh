#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p data/logs

# Run the game with improved veto settings
python -m src.game.main \
    --veto_mechanism uncertainty \
    --experiment_id adaptive_veto \
    --veto_timeout 15 \
    --uncertainty_threshold 0.25

# Run analysis on the data
python -m src.experiment.analyze --data_dir data/logs --verbose

# Generate visualizations
python -m src.experiment.visualize --data_dir data/logs --output_dir data/viz --all

echo "Done! Analysis and visualizations are available in data/logs/analysis and data/viz" 