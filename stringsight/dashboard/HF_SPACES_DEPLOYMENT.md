# Deploy to HuggingFace Spaces

## Quick Deploy

```bash
# Setup (one time)
pip install huggingface_hub[cli]
huggingface-cli login

# Install Git LFS (required for large files >10MB)
# Ubuntu/Debian: sudo apt-get install git-lfs
# macOS: brew install git-lfs

# Deploy (creates new Space or updates existing one)
python -m stringsight.dashboard.deploy_to_hf \
    --results_dir /path/to/results \
    --space_name my-dashboard \
    --hf_username your-username \
    --push
```

View at: `https://huggingface.co/spaces/your-username/my-dashboard`

**Note:** The `--push` flag automatically creates the Space if it doesn't exist, or updates it if it already exists.

## Common Options

**Private Space:**
```bash
python -m stringsight.dashboard.deploy_to_hf \
    --results_dir ./results \
    --space_name my-dashboard \
    --hf_username your-username \
    --private \
    --push
```

**Multiple Experiments:**
```bash
# If results/ contains multiple experiment subdirectories
python -m stringsight.dashboard.deploy_to_hf \
    --results_dir ./results \
    --space_name my-dashboard \
    --hf_username your-username \
    --push
```

**Create Files Only (Manual Push):**
```bash
python -m stringsight.dashboard.deploy_to_hf \
    --results_dir ./results \
    --space_name my-dashboard \
    --hf_username your-username
# Files created in ./hf_space_my-dashboard/
```

## Python API

```python
from stringsight.dashboard import deploy_to_hf_spaces

deploy_to_hf_spaces(
    results_dir="./results/my_experiment",
    space_name="my-dashboard",
    hf_username="your-username",
    push=True
)
```

## Required Results Structure

Your results directory must contain:
- `model_cluster_scores.json`
- `cluster_scores.json`
- `model_scores.json`
- `clustered_results_lightweight.jsonl`

**Note:** Only these 4 files are copied to the Space (other files like embeddings, full clustered results, etc. are excluded to minimize Space size and stay within the 1GB HuggingFace limit)

## Update Existing Space

```bash
cd hf_space_my-dashboard
cp -r /path/to/new/results/* results/
git add .
git commit -m "Update results"
git push
```

## Troubleshooting

**"Results directory does not contain required files"**
- Check that your results folder has the required JSON files above

**Git push fails**
- Run: `huggingface-cli login`

**Space not found**
- Create manually at https://huggingface.co/new-space first

**Get help:**
```bash
python -m stringsight.dashboard.deploy_to_hf --help
```
