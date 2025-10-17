/**
 * Utility functions for metrics data processing.
 * 
 * Handles the sanitization and matching of quality metric names
 * between frontend display and backend storage.
 */

/**
 * Sanitize metric name to match backend format.
 * Converts "omni_math_accuracy (0/1)" to "omni_math_accuracy_0_1"
 */
export function sanitizeMetricName(metricName: string): string {
  return metricName
    .replace(/\s+/g, '_')
    .replace(/[()]/g, '')
    .replace(/\//g, '_')
    .replace(/-/g, '_');
}

/**
 * Find the best matching metric in available metrics list.
 * Handles both sanitized and original formats.
 */
export function findMatchingMetric(targetMetric: string, availableMetrics: string[]): string | null {
  // Direct match first
  if (availableMetrics.includes(targetMetric)) {
    return targetMetric;
  }
  
  // Try sanitized version
  const sanitized = sanitizeMetricName(targetMetric);
  if (availableMetrics.includes(sanitized)) {
    return sanitized;
  }
  
  // Try reverse - check if any available metric sanitizes to target
  for (const metric of availableMetrics) {
    if (sanitizeMetricName(metric) === targetMetric) {
      return metric;
    }
  }
  
  return null;
}

/**
 * Get display name for a metric (reverse of sanitization for UI).
 */
export function getDisplayName(metricName: string): string {
  // Convert underscores back to spaces and add parentheses for ratios
  return metricName
    .replace(/_(\d+)_(\d+)$/, ' ($1/$2)')  // Convert _0_1 to (0/1)
    .replace(/_/g, ' ');
}

/**
 * Get original metric name from sanitized name (for data lookup).
 * This reverses the sanitization but keeps underscores in the base name.
 */
export function getOriginalMetricName(sanitizedName: string): string {
  // Convert _0_1 pattern back to (0/1) but keep other underscores
  return sanitizedName.replace(/_(\d+)_(\d+)$/, ' ($1/$2)');
}

/**
 * Validate if a quality metric exists in the data with either name format.
 */
export function hasQualityMetric(data: any[], metricName: string): boolean {
  if (!data.length) return false;
  
  const sampleRow = data[0];
  const qualityKey = `quality_${metricName}`;
  const sanitizedKey = `quality_${sanitizeMetricName(metricName)}`;
  
  return qualityKey in sampleRow || sanitizedKey in sampleRow;
}

export default {
  sanitizeMetricName,
  findMatchingMetric,
  getDisplayName,
  getOriginalMetricName,
  hasQualityMetric
};
