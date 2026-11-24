"""Read-only environment inspection. No installation, downloads or training."""
import importlib.metadata
import json,platform,sys

report=dict(python=sys.version,platform=platform.platform(),architecture=platform.machine())
for package in ['numpy','scipy','soundfile','scikit-learn','torch']:
    try:report[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:report[package]=None
try:
    import torch
    report.update(cuda_available=torch.cuda.is_available(),torch_cuda_runtime=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except ImportError:report['cuda_available']=False
print(json.dumps(report,ensure_ascii=False,indent=2))
if not report['cuda_available']:
    print('Training requires the Spark CUDA PyTorch environment. Metadata preparation can still run without CUDA.')
