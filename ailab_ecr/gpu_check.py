"""Stronger version of Juan's gpu_test.py.
'CUDA is available' can be True even when PyTorch has no kernels for the GPU,
so this also runs real GPU operations, including the graph layer ECR uses."""
import torch

print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("device:", torch.cuda.get_device_name(0), "| capability sm_%d%d" % cap)
    print("compiled arch list:", torch.cuda.get_arch_list())
    x = torch.randn(512, 512, device="cuda")
    print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
    from torch_geometric.nn import RGCNConv
    conv = RGCNConv(16, 16, num_relations=3, num_bases=2).cuda()
    ei = torch.randint(0, 50, (2, 200), device="cuda"); et = torch.randint(0, 3, (200,), device="cuda")
    print("RGCN (ECR graph layer) ok:", tuple(conv(torch.randn(50, 16, device="cuda"), ei, et).shape))
    import torch_sparse
    print("torch_sparse", torch_sparse.__version__, "ok")
