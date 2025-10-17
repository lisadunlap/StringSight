# HuggingFace Spaces Deployment Guide

This guide will help you deploy the Benchmark Results Viewer to HuggingFace Spaces.

## Prerequisites

1. A HuggingFace account (free at https://huggingface.co)
2. Your benchmark results in JSONL format
3. Git and Git LFS installed

## Step 1: Prepare Your Data

1. Create a `results/` directory in this folder:
   ```bash
   mkdir -p results
   ```

2. Copy your benchmark results into this directory with the following structure:
   ```
   results/
   ├── dataset_name/
   │   ├── behavior1.jsonl
   │   ├── behavior2.jsonl
   │   └── behavior3.jsonl
   ```

3. Each JSONL file should contain one JSON object per line with these fields:
   - `prompt`: The input prompt (required)
   - `model_response`: The model's response (required)
   - `system_prompt`: System prompt used (required)
   - `category`: Behavior category (required)
   - `behavior_description`: Description of the behavior (required)
   - `model`: Model identifier (required)

## Step 2: Create a HuggingFace Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in the details:
   - **Name**: Choose a name (e.g., "benchmark-results-viewer")
   - **License**: Choose an appropriate license
   - **SDK**: Select "Gradio"
   - **Visibility**: Public or Private

## Step 3: Clone the Space Repository

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
```

## Step 4: Copy Files to the Space

```bash
# From the huggingface_space directory, copy all files
cp app.py YOUR_SPACE_DIRECTORY/
cp requirements.txt YOUR_SPACE_DIRECTORY/
cp README.md YOUR_SPACE_DIRECTORY/
cp -r results/ YOUR_SPACE_DIRECTORY/
```

## Step 5: Push to HuggingFace

```bash
cd YOUR_SPACE_DIRECTORY

# Add all files
git add .

# Commit
git commit -m "Initial commit: Benchmark Results Viewer"

# Push to HuggingFace
git push
```

## Step 6: Wait for Build

Once pushed, HuggingFace will automatically:
1. Install dependencies from `requirements.txt`
2. Build the Gradio app
3. Deploy it to a public URL

This typically takes 1-3 minutes. You'll see the status at the top of your Space page.

## Step 7: Access Your Space

Your Space will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

## Updating Your Space

To update your Space with new data or code:

```bash
cd YOUR_SPACE_DIRECTORY

# Make your changes (e.g., add new results)
cp -r /path/to/new/results/* results/

# Commit and push
git add .
git commit -m "Update results with new data"
git push
```

## Tips

1. **Large Files**: If your results are large (>10MB), use Git LFS:
   ```bash
   git lfs install
   git lfs track "results/**/*.jsonl"
   git add .gitattributes
   ```

2. **Memory Issues**: If you have many results, consider:
   - Splitting into multiple smaller Spaces
   - Using the private Space option
   - Upgrading to a paid HuggingFace plan for more resources

3. **Custom Domain**: Paid HuggingFace plans allow custom domains

4. **Private Spaces**: Make your Space private in settings if needed

## Troubleshooting

### Space won't build
- Check the build logs at the bottom of your Space page
- Verify `requirements.txt` has correct package names
- Ensure `app.py` has no syntax errors

### No data showing
- Verify the `results/` directory structure matches the expected format
- Check that JSONL files have the required fields
- Look at the browser console for error messages

### Out of memory
- Reduce the number of examples
- Use smaller datasets
- Consider upgrading to a paid plan

## Support

For issues specific to HuggingFace Spaces, see:
- HuggingFace Spaces Documentation: https://huggingface.co/docs/hub/spaces
- Community Forum: https://discuss.huggingface.co/


