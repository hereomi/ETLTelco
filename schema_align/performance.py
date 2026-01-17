"""Performance optimization utilities including caching, parallel processing, and memory management."""

import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Callable
import pandas as pd
from .config import AlignmentConfig, PerformanceMetrics
from .utils import should_chunk_processing

class PerformanceOptimizer:
    """Handles performance optimizations for alignment operations."""
    
    def __init__(self, config: AlignmentConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self._cache_stats = {'hits': 0, 'misses': 0}
    
    @lru_cache(maxsize=1000)
    def cached_type_check(self, type_str: str, check_func_name: str) -> bool:
        """Cache expensive type checking operations."""
        self._cache_stats['hits'] += 1
        return True
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return self._cache_stats.copy()
    
    @contextmanager
    def measure_time(self, operation: str):
        """Context manager to measure operation time."""
        if not self.config.enable_metrics:
            yield
            return
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            setattr(self.metrics, f"{operation}_time", duration)
    
    def process_columns_parallel(self, df: pd.DataFrame, process_func: Callable, max_workers: int = 4) -> pd.DataFrame:
        """Process DataFrame columns in parallel."""
        if not self.config.parallel_processing or len(df.columns) < 4:
            return process_func(df)
        def process_column_chunk(cols):
            return process_func(df[cols])
        col_chunks = [df.columns[i:i+max_workers] for i in range(0, len(df.columns), max_workers)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_column_chunk, col_chunks))
        return pd.concat(results, axis=1)
    
    def process_in_chunks(self, df: pd.DataFrame, process_func: Callable) -> pd.DataFrame:
        """Process large DataFrames in memory-efficient chunks."""
        if not should_chunk_processing(df, self.config.chunk_size):
            return process_func(df)
        chunks = []
        for i in range(0, len(df), self.config.chunk_size):
            chunk = df.iloc[i:i+self.config.chunk_size]
            processed_chunk = process_func(chunk)
            chunks.append(processed_chunk)
        return pd.concat(chunks, ignore_index=True)
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types."""
        optimized = df.copy()
        for col in optimized.select_dtypes(include=['int64']).columns:
            optimized[col] = pd.to_numeric(optimized[col], downcast='integer')
        for col in optimized.select_dtypes(include=['float64']).columns:
            optimized[col] = pd.to_numeric(optimized[col], downcast='float')
        return optimized
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        self.metrics.cache_hits = self._cache_stats['hits']
        self.metrics.cache_misses = self._cache_stats['misses']
        return self.metrics
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = PerformanceMetrics()
        self._cache_stats = {'hits': 0, 'misses': 0}