# Quick Start Guide

This is a streamlined Benchmark Results Viewer designed for HuggingFace Spaces deployment. The validation results tab has been removed to focus on browsing and comparing examples.

## What Changed

**Removed:**
- ✓ Validation Results tab (metrics, co-occurrence plots, validation examples)
- Plotly dependencies (no longer needed)
- Validation-related functions

**Kept:**
- All Behaviors tab - View all behavior responses for the same prompt
- System Prompts tab - View system prompts used for each behavior
- Behavior Overview tab - Detailed information about each behavior
- Browse Examples tab - Browse individual examples
- Compare Behaviors tab - Side-by-side comparison

## Quick Deployment (3 Steps)

### 1. Prepare Your Data

```bash
cd /home/lisabdunlap/StringSight/benchmark/huggingface_space

# Copy your benchmark results
./prepare_for_deployment.sh ../results
```

### 2. Test Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py --results-dir results/

# Open http://localhost:7860 in your browser
```

### 3. Deploy to HuggingFace

See `DEPLOYMENT_GUIDE.md` for detailed instructions on uploading to HuggingFace Spaces.

## File Structure

```
huggingface_space/
├── app.py                      # Main Gradio application
├── requirements.txt            # Python dependencies
├── README.md                   # Space description for HuggingFace
├── DEPLOYMENT_GUIDE.md         # Detailed deployment instructions
├── QUICK_START.md             # This file
├── prepare_for_deployment.sh  # Script to prepare data
├── .gitignore                 # Git ignore file
└── results/                   # Your benchmark results go here
    ├── dataset_1/
    │   ├── behavior1.jsonl
    │   └── behavior2.jsonl
    └── dataset_2/
        └── behavior1.jsonl
```

## Dependencies

Minimal dependencies for easy deployment:
- `gradio>=4.0.0` - For the UI
- `pandas>=2.0.0` - For data handling

No heavy dependencies like plotly or validation frameworks.

## Features

### All Behaviors Tab
- View all behavior responses for the same prompt
- Navigate through examples with a slider
- See all behaviors side-by-side

### System Prompts Tab
- View base system prompt
- See modifications for each behavior
- Understand how behaviors are induced

### Behavior Overview Tab
- Select a dataset and behavior
- View category, description, and full system prompt
- See number of examples

### Browse Examples Tab
- Select dataset and specific behavior
- Navigate through individual examples
- See prompt and response clearly

### Compare Behaviors Tab
- Select two behaviors to compare
- See how they respond to the same prompt
- Navigate through examples

## Notes

- All validation-related code has been removed to simplify the viewer
- The viewer now focuses purely on browsing and comparing examples
- Perfect for sharing benchmark results publicly
- Optimized for HuggingFace Spaces deployment

## Need Help?

1. Test locally first to ensure your data loads correctly
2. Check `results/README.md` for data format requirements
3. See `DEPLOYMENT_GUIDE.md` for deployment troubleshooting
4. Make sure your JSONL files have all required fields


