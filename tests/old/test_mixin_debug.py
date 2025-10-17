#!/usr/bin/env python3
"""
Debug test for mixin initialization issues.
"""

from stringsight.core.stage import PipelineStage
from stringsight.core.mixins import LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin

# Test each mixin individually first
print("Testing individual mixins...")

class TestLoggingOnly(PipelineStage, LoggingMixin):
    def run(self, data): return data

class TestTimingOnly(PipelineStage, TimingMixin):
    def run(self, data): return data

class TestErrorOnly(PipelineStage, ErrorHandlingMixin):
    def run(self, data): return data

class TestWandbOnly(PipelineStage, WandbMixin):
    def run(self, data): return data

# Test combinations
class TestTwoMixins(PipelineStage, LoggingMixin, TimingMixin):
    def run(self, data): return data

class TestThreeMixins(PipelineStage, LoggingMixin, TimingMixin, ErrorHandlingMixin):
    def run(self, data): return data

class TestAllMixins(PipelineStage, LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin):
    def run(self, data): return data

def test_mixin(cls, name, **kwargs):
    try:
        instance = cls(**kwargs)
        print(f"✅ {name} - SUCCESS")
        return True
    except Exception as e:
        print(f"❌ {name} - FAILED: {e}")
        return False

if __name__ == "__main__":
    # Test individual mixins
    test_mixin(TestLoggingOnly, "LoggingMixin only", verbose=True)
    test_mixin(TestTimingOnly, "TimingMixin only")
    test_mixin(TestErrorOnly, "ErrorHandlingMixin only", fail_fast=True)
    test_mixin(TestWandbOnly, "WandbMixin only", use_wandb=True, wandb_project="test")
    
    print("\nTesting combinations...")
    test_mixin(TestTwoMixins, "Logging + Timing", verbose=True)
    test_mixin(TestThreeMixins, "Logging + Timing + Error", verbose=True, fail_fast=True)
    test_mixin(TestAllMixins, "All mixins", verbose=True, fail_fast=True, use_wandb=True, wandb_project="test")
    
    print("\nTesting with extra kwargs...")
    test_mixin(TestAllMixins, "All mixins + extra kwargs", 
               verbose=True, fail_fast=True, use_wandb=True, wandb_project="test",
               extra_param="should_be_filtered", another_param=123) 