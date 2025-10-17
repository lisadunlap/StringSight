# API Reference

Complete API documentation for StringSight's main functions and classes.

## Main Entry Points

### explain()

::: stringsight.public.explain
    options:
      show_root_heading: true
      heading_level: 3

### label()

::: stringsight.public.label
    options:
      show_root_heading: true
      heading_level: 3

### extract_properties_only()

::: stringsight.public.extract_properties_only
    options:
      show_root_heading: true
      heading_level: 3

### compute_metrics_only()

::: stringsight.public.compute_metrics_only
    options:
      show_root_heading: true
      heading_level: 3

## Convenience Functions

### explain_side_by_side()

::: stringsight.public.explain_side_by_side
    options:
      show_root_heading: true
      heading_level: 3

### explain_single_model()

::: stringsight.public.explain_single_model
    options:
      show_root_heading: true
      heading_level: 3

### explain_with_custom_pipeline()

::: stringsight.public.explain_with_custom_pipeline
    options:
      show_root_heading: true
      heading_level: 3

## Core Data Structures

### PropertyDataset

::: stringsight.core.data_objects.PropertyDataset
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - from_dataframe
        - save
        - load
        - to_dataframe

### ConversationRecord

::: stringsight.core.data_objects.ConversationRecord
    options:
      show_root_heading: true
      heading_level: 3

### Property

::: stringsight.core.data_objects.Property
    options:
      show_root_heading: true
      heading_level: 3

### Cluster

::: stringsight.core.data_objects.Cluster
    options:
      show_root_heading: true
      heading_level: 3

## Pipeline Components

### PipelineStage

::: stringsight.core.stage.PipelineStage
    options:
      show_root_heading: true
      heading_level: 3

### Pipeline

::: stringsight.pipeline.Pipeline
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - run
        - add_stage

## Extractors

### get_extractor()

::: stringsight.extractors.get_extractor
    options:
      show_root_heading: true
      heading_level: 3

### OpenAIExtractor

::: stringsight.extractors.openai.OpenAIExtractor
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - run

## Clusterers

### get_clusterer()

::: stringsight.clusterers.get_clusterer
    options:
      show_root_heading: true
      heading_level: 3

### HDBSCANClusterer

::: stringsight.clusterers.hdbscan.HDBSCANClusterer
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - run

## Metrics

### get_metrics()

::: stringsight.metrics.get_metrics
    options:
      show_root_heading: true
      heading_level: 3

## Utilities

### Format Detection

::: stringsight.formatters.detect_method
    options:
      show_root_heading: true
      heading_level: 3

### Validation

::: stringsight.formatters.validate_required_columns
    options:
      show_root_heading: true
      heading_level: 3
