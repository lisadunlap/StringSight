# Release Workflow & Branch Strategy

## Branch Structure

StringSight uses a **two-branch strategy** to separate development from production releases:

### `main` Branch (Development)
- **Purpose:** Active development with all data, tests, and experiments
- **Contains:**
  - Full `data/` directory with example datasets
  - `results/` and `benchmark/results/` with evaluation outputs
  - `notebooks/` for experiments
  - Development tools and debug files
  - All git LFS tracked data files

### `prod` Branch (Production/Release)
- **Purpose:** Clean, minimal codebase for PyPI releases
- **Contains:**
  - Only `stringsight/` package code
  - Essential documentation (README, LICENSE, CHANGELOG)
  - Frontend and benchmark code
  - Scripts and examples
- **Excludes:**
  - Data directories (`data/`, `results/`, `benchmark/results/`)
  - Large files (`google-cloud-sdk/`, `*.tar.gz`)
  - Debug files (`playground.ipynb`, test scripts)
  - Internal documentation (CLAUDE.md, README_ABSTRACTION.md)
  - Demo/backup code (`api_backup/`, `autoformat_demo/`)

## Working with Branches

### Development Workflow (main)

```bash
# Work on main branch
git checkout main

# Your data is here
ls data/           # ✅ Full data directory
ls notebooks/      # ✅ Notebooks available

# Make changes, commit normally
git add .
git commit -m "Add new feature"
git push origin main
```

### Release Workflow (prod)

```bash
# 1. Update version in pyproject.toml on main
git checkout main
# Edit version in pyproject.toml: 0.1.1 -> 0.1.2
git commit -am "Bump version to 0.1.2"

# 2. Merge main into prod (carefully!)
git checkout prod
git merge main --no-commit

# 3. Verify no unwanted files were added
git status
# If any data/ or results/ files appear, remove them:
# git rm -rf data/ results/ benchmark/results/

# 4. Commit the merge
git commit -m "Merge main for release v0.1.2"

# 5. Build and publish
python -m build
twine upload dist/*

# 6. Tag the release
git tag -a v0.1.2 -m "Release version 0.1.2"
git push origin prod --tags

# 7. Switch back to main for development
git checkout main
```

## Why This Strategy?

### Benefits:
1. **Smaller Git Clones:** Users cloning `prod` don't download GB of data
2. **Clean PyPI Packages:** Built from prod, no data files included
3. **Development Freedom:** Keep all data/experiments in main without bloat
4. **Separate History:** prod has minimal commit history

### Important Notes:

⚠️ **Never merge prod → main** (would delete data!)  
✅ **Always merge main → prod** (and clean up after)  
✅ **Always build releases from prod branch**  
✅ **Keep developing on main with all your data**

## Syncing Between Branches

When you add code changes on main that should go to prod:

```bash
# On main: make your changes
git checkout main
# ... make changes ...
git commit -am "Add new feature"
git push origin main

# Merge to prod
git checkout prod
git merge main --no-commit

# Clean up any unwanted files that snuck in
git rm -rf data/ results/ notebooks/ || true

# Commit and push
git commit -m "Sync: Add new feature from main"
git push origin prod
```

## Initial Setup (Already Done)

This was the one-time setup that created the prod branch:

```bash
# 1. Update .gitignore on main
# (Already done - prevents re-adding data/)

# 2. Create prod branch from main
git checkout -b prod

# 3. Remove data and large files
git rm -rf data/ results/ benchmark/results/ benchmark/evaluation_results/
git rm -rf notebooks/
git rm config.yaml mast.json playground.ipynb
git rm CLAUDE.md README_ABSTRACTION.md
git rm -rf stringsight/api_backup/ stringsight/autoformat_demo/
git rm setup.py

# 4. Commit
git commit -m "Clean prod branch for release"

# 5. Push both branches
git push origin main
git push origin prod
```

## Troubleshooting

### "I accidentally committed data to prod!"

```bash
git checkout prod
git rm -rf data/ results/ benchmark/results/
git commit --amend
# Or if already pushed:
git commit -m "Remove data files from prod"
git push origin prod
```

### "I need to see what's different between branches"

```bash
# Compare what prod has that main doesn't
git diff main..prod --name-status | grep "^D" | head -20

# Compare what main has that prod doesn't  
git diff prod..main --name-status | grep "^A" | head -20
```

### "I want to test the package before publishing"

```bash
git checkout prod
python -m build
pip install dist/stringsight-0.1.2-py3-none-any.whl
# Test it...
pip uninstall stringsight
```

## Package Publishing Checklist

Before each release from `prod`:

- [ ] Version updated in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] All tests passing on main
- [ ] Changes merged from main to prod
- [ ] No data/ directories in prod
- [ ] No large files (google-cloud-sdk/, *.tar.gz)
- [ ] Build succeeds: `python -m build`
- [ ] Package tested locally
- [ ] Published to PyPI: `twine upload dist/*`
- [ ] Tagged: `git tag -a v0.1.2 -m "Release 0.1.2"`
- [ ] Tags pushed: `git push --tags`

---

**Current Status:**
- ✅ `main` branch: Full development environment with data
- ✅ `prod` branch: Clean release-ready codebase
- ✅ `.gitignore` updated to prevent re-adding data
- ✅ Ready for release workflow

