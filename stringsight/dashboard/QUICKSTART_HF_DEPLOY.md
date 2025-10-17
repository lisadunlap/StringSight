# Quick Deploy to HuggingFace Spaces

## One Command Deploy

```bash
# Setup (one time)
pip install huggingface_hub[cli]
huggingface-cli login

# Install Git LFS (required for large files)
# Ubuntu/Debian: sudo apt-get install git-lfs
# macOS: brew install git-lfs

# Deploy
python -m stringsight.dashboard.deploy_to_hf \
    --results_dir /path/to/results \
    --space_name my-dashboard \
    --hf_username your-username \
    --push
```

**Done!** View at: `https://huggingface.co/spaces/your-username/my-dashboard`

## Options

```bash
# Private Space
--private

# Manual review before push (omit --push)
python -m stringsight.dashboard.deploy_to_hf \
    --results_dir ./results \
    --space_name my-dashboard \
    --hf_username your-username

# Get help
--help
```

## Python API

```python
from stringsight.dashboard import deploy_to_hf_spaces

deploy_to_hf_spaces(
    results_dir="./results",
    space_name="my-dashboard",
    hf_username="your-username",
    push=True
)
```

Full docs: [HF_SPACES_DEPLOYMENT.md](HF_SPACES_DEPLOYMENT.md)
