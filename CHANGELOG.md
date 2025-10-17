# Changelog

All notable changes to StringSight will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-10-09

### Added
- `sample_size` parameter to `explain()` and `label()` functions for built-in dataset sampling
- Intelligent sampling that automatically handles balanced datasets and side-by-side comparisons
- Documentation updates across README and user guides for the new parameter

### Changed
- Simplified workflow: users can now pass `sample_size` directly instead of manually subsampling

## [0.1.1] - Previous Release

### Added
- Comprehensive documentation site with MkDocs
- Auto-generated API reference using mkdocstrings
- User guides for data formats, visualization, and configuration
- Advanced guides for custom pipelines and performance tuning
- React frontend for interactive analysis
- Streamlit dashboard with multi-tab interface
- Gradio chat viewer for conversation inspection
- Fixed-taxonomy labeling with `label()` function
- Task-aware analysis with `task_description` parameter
- Bootstrap confidence intervals for metrics
- Multi-dimensional quality scoring
- Hierarchical clustering (fine/coarse levels)
- Cost tracking for LLM API usage
- W&B integration for experiment tracking
- OpenAI conversation format support (tool calls, multimodal)
- Automatic format detection and conversion
- Caching at extraction, clustering, and metrics stages
- Side-by-side conversion from tidy single-model data

### Changed
- Project renamed from "StringSight" to "StringSight"
- Updated all documentation to use consistent naming
- Modernized mkdocs.yml configuration with Material theme
- Improved README with professional formatting
- Consolidated metric calculations into functional pipeline
- Enhanced error messages and validation

### Fixed
- CORS configuration for frontend/backend communication
- OpenAI conversation format parsing edge cases
- Cluster labeling with hierarchical clustering
- Bootstrap sampling for small datasets
- Memory usage with large embedding matrices

### Removed
- Outdated/fictional API documentation pages
- Placeholder content from docs/
- Weave tracing (replaced with W&B only)

## [0.1.0] - Initial Release

### Added
- Core pipeline architecture (extraction → post-processing → clustering → metrics)
- `explain()` function for main analysis workflow
- Single-model and side-by-side analysis modes
- OpenAI and vLLM extractor support
- HDBSCAN and hierarchical clustering
- Property extraction with LLM analysis
- Basic visualization support
- Command-line interface via scripts/

---

## Migration Notes

### From StringSight to StringSight

**Import Changes:**
```python
# Old
from lmm_vibes import explain

# New
from stringsight import explain
```

**No other breaking changes** - All function signatures and parameters remain the same.

### Future Breaking Changes

None planned for v1.0 release.
