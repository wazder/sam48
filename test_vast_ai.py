#!/usr/bin/env python3
"""
Test script for vast.ai RTX A4000 environment
Validates GPU availability, downloads test video, runs Sam48 pipeline
"""
import os
import sys
import torch
import subprocess
from loguru import logger
import time
import psutil

def check_system_info():
    """Check basic system information"""
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # System resources
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    memory = psutil.virtual_memory()
    logger.info(f"RAM: {memory.total / 1024**3:.1f} GB (Available: {memory.available / 1024**3:.1f} GB)")

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("=== Checking Dependencies ===")
    
    required_packages = [
        'torch', 'torchvision', 'ultralytics', 'opencv-python', 
        'numpy', 'loguru', 'pyyaml', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} - OK")
        except ImportError:
            logger.error(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def download_models():
    """Download required models if not present"""
    logger.info("=== Checking Models ===")
    
    # Check YOLO model
    yolo_model = "yolov8n.pt"
    if not os.path.exists(yolo_model):
        logger.info(f"Downloading {yolo_model}...")
        try:
            from ultralytics import YOLO
            model = YOLO(yolo_model)  # This will download the model
            logger.success(f"✅ {yolo_model} downloaded")
        except Exception as e:
            logger.error(f"❌ Failed to download {yolo_model}: {e}")
            return False
    else:
        logger.info(f"✅ {yolo_model} already exists")
    
    # Check SAM model
    sam_model = "sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_model):
        logger.warning(f"⚠️  {sam_model} not found - will need to be downloaded separately")
        logger.info("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    else:
        logger.info(f"✅ {sam_model} already exists")
    
    return True

def test_gpu_memory():
    """Test GPU memory availability"""
    if not torch.cuda.is_available():
        logger.warning("No GPU available for memory test")
        return True
    
    logger.info("=== GPU Memory Test ===")
    
    try:
        device = torch.device('cuda:0')
        
        # Check initial memory
        initial_memory = torch.cuda.memory_allocated(device) / 1024**3
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        logger.info(f"Initial GPU memory: {initial_memory:.2f} GB / {total_memory:.2f} GB")
        
        # Test allocation
        test_tensor = torch.randn(1000, 1000, 1000, device=device)
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
        
        logger.info(f"After allocation: {allocated_memory:.2f} GB / {total_memory:.2f} GB")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"After cleanup: {final_memory:.2f} GB / {total_memory:.2f} GB")
        
        logger.success("✅ GPU memory test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU memory test failed: {e}")
        return False

def download_test_video():
    """Download test video from Google Drive"""
    logger.info("=== Downloading Test Video ===")
    
    try:
        result = subprocess.run([sys.executable, "download_test_video.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.success("✅ Test video downloaded successfully")
            return True
        else:
            logger.error(f"❌ Video download failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Video download timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Video download error: {e}")
        return False

def run_pipeline_test():
    """Run the Sam48 pipeline on test video"""
    logger.info("=== Running Pipeline Test ===")
    
    test_video = "test_videos/test_video.mp4"
    if not os.path.exists(test_video):
        logger.error(f"❌ Test video not found: {test_video}")
        return False
    
    # Create test output directory
    test_output = "test_output"
    os.makedirs(test_output, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Run pipeline
        cmd = [sys.executable, "main.py", test_video, "--output", test_output]
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result.returncode == 0:
            logger.success(f"✅ Pipeline test completed successfully in {processing_time:.1f}s")
            
            # Check outputs
            output_files = []
            for root, dirs, files in os.walk(test_output):
                for file in files:
                    output_files.append(os.path.join(root, file))
            
            logger.info(f"Generated {len(output_files)} output files:")
            for file in output_files[:10]:  # Show first 10 files
                logger.info(f"  - {file}")
            
            if len(output_files) > 10:
                logger.info(f"  ... and {len(output_files) - 10} more files")
            
            return True
        else:
            logger.error(f"❌ Pipeline test failed: {result.stderr}")
            logger.error(f"stdout: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Pipeline test timed out (30 minutes)")
        return False
    except Exception as e:
        logger.error(f"❌ Pipeline test error: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting vast.ai RTX A4000 Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("System Info", check_system_info),
        ("Dependencies", check_dependencies),
        ("Models", download_models),
        ("GPU Memory", test_gpu_memory),
        ("Download Video", download_test_video),
        ("Pipeline Test", run_pipeline_test)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        if test_name == "System Info":
            test_func()  # This test always passes, just shows info
            passed_tests += 1
        else:
            try:
                if test_func():
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} test failed")
            except Exception as e:
                logger.error(f"❌ {test_name} test crashed: {e}")
    
    # Final results
    logger.info("\n" + "=" * 50)
    logger.info("🏁 Test Suite Complete")
    logger.info(f"Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.success("🎉 All tests passed! Your vast.ai setup is ready.")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Check logs above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)