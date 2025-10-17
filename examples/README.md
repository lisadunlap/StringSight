# StringSight Examples

## `deploy_dashboard_to_hf.py`

Deploy your dashboard to HuggingFace Spaces:

```bash
python examples/deploy_dashboard_to_hf.py
```

Or use CLI directly:
```bash
python -m stringsight.dashboard.deploy_to_hf \
    --results_dir ./results \
    --space_name my-dashboard \
    --hf_username your-username \
    --push
```

See: [HF_SPACES_DEPLOYMENT.md](../stringsight/dashboard/HF_SPACES_DEPLOYMENT.md)
