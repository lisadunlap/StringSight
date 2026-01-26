# Performance Analysis Guide

## Quick Start - Find Where Time is Being Spent

### 1. Run the test and see timing breakdown:
```bash
python test_timing_instrumentation.py 2>&1 | grep '\[TIMING\]'
```

**Expected Output:**
```
[TIMING] /extract/single request started
[TIMING] Dynamic prompt generation started
[TIMING] Task expansion completed in 3.241s
[TIMING] Discovery generation (3 parallel LLM calls) completed in 4.103s
[TIMING] Verification attempt 1 completed in 2.876s
[TIMING] Verification + reflection total: 2.876s (reflection_time: 0.000s, attempts: 0)
[TIMING] Clustering customization (3 parallel LLM calls) completed in 3.521s
[TIMING] generate_all_prompts() completed in 13.741s
[TIMING] Dynamic prompt generation total: 13.741s (reflection_attempts: 0)
[TIMING] Prompt generation completed in 13.741s
[TIMING] Property extraction completed in 2.105s
[TIMING] /extract/single total: 15.891s (prompts: 13.7s, extraction: 2.1s)
```

### 2. Check for retry/rate limit issues:
```bash
python test_timing_instrumentation.py 2>&1 | grep -E '\[TIMING\] Retry|Rate limit'
```

If you see output like:
```
[TIMING] Retry 1/3 sleeping 5.0s (reason: RateLimitError)
[TIMING] Retry 2/3 sleeping 10.0s (reason: RateLimitError)
```

**This is your problem!** Retries are adding 15-35 seconds per failed call.

### 3. See individual LLM call durations (DEBUG mode):
```bash
STRINGSIGHT_LOG_LEVEL=DEBUG python test_timing_instrumentation.py 2>&1 | grep '\[TIMING\] LLM API'
```

**Expected Output:**
```
[TIMING] LLM API call completed in 2.341s
[TIMING] LLM API call completed in 1.987s
[TIMING] LLM API call completed in 3.124s
```

If any individual call takes >10s, the API itself is slow.

### 4. Check cache effectiveness:
```bash
STRINGSIGHT_LOG_LEVEL=DEBUG python test_timing_instrumentation.py 2>&1 | grep 'Cache'
```

**Good output (caching working):**
```
[TIMING] Cache hit in 0.1ms
[TIMING] Cache hit in 0.1ms
Cache HIT (memory): completion:a3b2c1...
```

**Bad output (cache not working):**
```
[TIMING] Cache miss in 0.2ms
Cache MISS: completion:a3b2c1...
```

### 5. Save full log for analysis:
```bash
STRINGSIGHT_LOG_LEVEL=DEBUG python test_timing_instrumentation.py 2>&1 | tee timing_full.log
```

Then analyze:
```bash
# All timing info
grep '\[TIMING\]' timing_full.log

# Find slow operations (>5s)
grep '\[TIMING\].*[5-9]\.' timing_full.log

# Check for errors
grep -i 'error\|exception\|failed' timing_full.log
```

## Interpreting Results

### Scenario 1: Slow Task Expansion (>10s)
```
[TIMING] Task expansion completed in 18.926s  ← PROBLEM HERE
```

**Possible causes:**
- First LLM call is slow (cold start)
- Task expansion hitting rate limits
- Task expansion LLM call is complex (large context)

**To debug:**
```bash
STRINGSIGHT_LOG_LEVEL=DEBUG python test_timing_instrumentation.py 2>&1 | grep -A5 "Task expansion"
```

### Scenario 2: Slow Discovery Generation (>10s)
```
[TIMING] Discovery generation (3 parallel LLM calls) completed in 15.123s  ← PROBLEM
```

**Possible causes:**
- One of the 3 parallel calls is slow
- Rate limiting slowing down parallel calls
- Discovery generator hitting API limits

**To debug:**
```bash
# Look for which call in the trio is slow
STRINGSIGHT_LOG_LEVEL=DEBUG python test_timing_instrumentation.py 2>&1 | grep -B2 -A2 "Discovery generation"
```

### Scenario 3: Slow Verification (>10s per attempt)
```
[TIMING] Verification attempt 1 completed in 12.543s  ← PROBLEM
[TIMING] Verification attempt 2 completed in 11.234s
```

**Possible causes:**
- Verification LLM call is slow
- Large prompt being verified
- Rate limiting

### Scenario 4: Multiple Reflection Attempts
```
[TIMING] Verification + reflection total: 45.123s (reflection_time: 25.000s, attempts: 3)  ← PROBLEM
```

**This means:**
- Verification failed 3 times
- Each reflection triggered an LLM call to fix the prompt
- Each re-verification tested the fixed prompt

**Why this happens:**
- Generated prompt produces invalid JSON
- Prompt format doesn't match expectations
- LLM is being too creative with the output format

### Scenario 5: Retry Hell
```
[TIMING] Retry 1/3 sleeping 5.0s (reason: RateLimitError)
[TIMING] Retry 2/3 sleeping 10.0s (reason: RateLimitError)
[TIMING] Retry 3/3 sleeping 20.0s (reason: RateLimitError)
```

**This is the main cause of 1-2 minute delays!**

- Each failed call sleeps for 5s, 10s, 20s (total: 35s)
- With 8-14 LLM calls, if half fail → 140-245 seconds extra
- Rate limiting is the most common cause

**Solutions:**
1. Reduce API request rate (fewer parallel calls)
2. Use a different API key with higher rate limits
3. Add exponential backoff starting time to environment
4. Reduce `base_sleep_time` from 5.0s to 2.0s

### Scenario 6: Cache Not Working
```
Cache MISS: completion:a3b2c1...
Cache MISS: completion:b4c5d6...
Cache MISS: completion:c5d6e7...
```

**Every call is cache miss = no caching benefit**

**Possible causes:**
- Different parameters each time (temperature, top_p, messages)
- Cache keys not deterministic
- Cache disabled via environment variable

**To check:**
```bash
# Verify cache is enabled
echo $STRINGSIGHT_DISABLE_CACHE  # Should be empty

# Run same request twice - second should be instant
python test_timing_instrumentation.py 2>&1 | grep 'total:'
python test_timing_instrumentation.py 2>&1 | grep 'total:'  # Should be <1s
```

## Performance Targets

### Good Performance:
- **First run (no cache):** 10-20 seconds total
  - Prompt generation: 8-15s
  - Property extraction: 2-5s
- **Second run (with cache):** <5 seconds total
  - Prompt generation: <1s (cached)
  - Property extraction: <1s (cached)

### Poor Performance (Your Current Issue):
- **First run:** 60-120 seconds total
  - Indicates: retries, rate limits, or reflection loops
- **Second run:** Still 60-120 seconds
  - Indicates: cache not working OR requests not identical

## Environment Variables Reference

Control logging verbosity:
```bash
# Basic timing (INFO level - default)
export STRINGSIGHT_LOG_LEVEL=INFO

# Detailed timing + cache info (DEBUG level)
export STRINGSIGHT_LOG_LEVEL=DEBUG

# Cache operation timing (only when needed)
export STRINGSIGHT_CACHE_DEBUG=1

# Disable caching (for testing)
export STRINGSIGHT_DISABLE_CACHE=1
```

## Commands for Your Specific Use Case

Since you're using `/extract/single` with dynamic prompts:

### 1. First, identify the bottleneck:
```bash
python test_timing_instrumentation.py 2>&1 | grep '\[TIMING\]'
```

Look at the final line:
```
[TIMING] /extract/single total: 62.3s (prompts: 58.1s, extraction: 4.2s)
```

If **prompts >> extraction**, the problem is in dynamic prompt generation.

### 2. Break down prompt generation:
```bash
python test_timing_instrumentation.py 2>&1 | grep '\[TIMING\]' | grep -E 'Task expansion|Discovery|Verification|Clustering'
```

This shows you which of the 4 steps is slow.

### 3. Check for retries (most likely cause):
```bash
python test_timing_instrumentation.py 2>&1 | grep '\[TIMING\] Retry'
```

If you see ANY retry logs, that's adding 5-35 seconds per retry.

### 4. Check individual LLM call times:
```bash
STRINGSIGHT_LOG_LEVEL=DEBUG python test_timing_instrumentation.py 2>&1 | grep '\[TIMING\] Single LLM request'
```

Output like:
```
[TIMING] Single LLM request completed in 15.234s (attempts: 3)
```

Means this call took 15s with 3 retry attempts (likely 2 failures).

## Next Steps

1. **Run the test** and share the `[TIMING]` output
2. **Identify which step** is taking 60+ seconds
3. **Look for retry patterns** - this is the most common cause
4. **Check if subsequent runs** are faster (cache working?)

Then we can optimize the specific bottleneck!
