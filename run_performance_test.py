#!/usr/bin/env python3
"""
Performance test with multiple configurations
"""
import time
import subprocess
import sys
import psutil
import threading
from loguru import logger

def monitor_resources():
    """Monitor CPU, GPU, Memory usage"""
    import GPUtil
    
    while monitor_running:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_util = gpu.load * 100
            gpu_memory = gpu.memoryUtil * 100
            
            logger.info(f"Resources - CPU: {cpu_percent:.1f}% | RAM: {memory_percent:.1f}% | GPU: {gpu_util:.1f}% | VRAM: {gpu_memory:.1f}%")
        
        time.sleep(5)

def run_performance_test():
    """Run pipeline with resource monitoring"""
    global monitor_running
    monitor_running = True
    
    # Start resource monitoring thread
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    logger.info("🚀 Starting performance test with optimized config")
    start_time = time.time()
    
    try:
        # Run pipeline
        result = subprocess.run([
            sys.executable, "main.py", 
            "test_videos/test_video.mp4", 
            "--output", "performance_test"
        ], capture_output=True, text=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        monitor_running = False
        
        if result.returncode == 0:
            logger.success(f"✅ Performance test completed in {processing_time:.1f}s")
            
            # Calculate FPS
            total_frames = 19649  # From video info
            fps = total_frames / processing_time
            logger.info(f"Processing FPS: {fps:.2f}")
            logger.info(f"Real-time factor: {fps/54.0:.2f}x")  # Video is 54fps
            
        else:
            logger.error("❌ Performance test failed")
            logger.error(f"Error: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        monitor_running = False

if __name__ == "__main__":
    # Install GPUtil if not present
    try:
        import GPUtil
    except ImportError:
        logger.info("Installing GPUtil...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gputil"])
        import GPUtil
    
    run_performance_test()