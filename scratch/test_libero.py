import os
try:
    from libero.libero import benchmark
    print("libero imported")
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    print("LIBERO-Spatial benchmark loaded successfully")
    print(task_suite.get_num_tasks())
except Exception as e:
    print(f"Error loading LIBERO: {e}")
