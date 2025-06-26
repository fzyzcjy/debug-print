import torch
import debug_print

debug_print.initialize()

print("demo without cuda graph...")
x = torch.rand(3, 4, 5).to(0)
debug_print.print_tensor(x)
debug_print.print_tensor(x[..., 0:3])
x = torch.arange(3 * 4 * 5, dtype=torch.int32).view(3, 4, 5).to(0)
debug_print.print_tensor(x[..., 0])
debug_print.print_tensor(x[0:1, 1:3, 0:4])

print("demo for all types...")
debug_print.print_tensor(torch.tensor([3, 4, 5], dtype=torch.int32, device="cuda:0"), name="for int32", print_shape=True, print_ptr=True)
debug_print.print_tensor(torch.tensor([3, 4, 5], dtype=torch.int64, device="cuda:0"), name="for int64", print_shape=True, print_ptr=True)
debug_print.print_tensor(torch.tensor([1.5, 2.5, 3.5], dtype=torch.float, device="cuda:0"), name="for float", print_shape=True, print_ptr=True)

print("demo for all dims...")
debug_print.print_tensor(torch.tensor([3, 4, 5], dtype=torch.int32, device="cuda:0"), name="for 1D", print_shape=True, print_ptr=True)
debug_print.print_tensor(torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.int32, device="cuda:0"), name="for 2D", print_shape=True, print_ptr=True)
debug_print.print_tensor(
    torch.tensor([[[1, 2, 3], [3, 4, 5]], [[10, 20, 30], [30, 40, 50]]], dtype=torch.int32, device="cuda:0"),
    name="for 3D", print_shape=True, print_ptr=True)

print("start warmup...")
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
x = torch.empty(2, 2).half().to(0)
y = torch.empty(2, 2).half().to(0)
with torch.cuda.stream(s):
    for i in range(3):
        z = x @ y
        z1 = z @ y
        z2 = z1 @ y

print("start graph capture...")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s):
    debug_print.print_tensor(x)
    debug_print.print_tensor(y, print_ptr=True)
    z = x @ y
    debug_print.print_tensor(z)
    z1 = z @ y
    debug_print.print_tensor(z1[..., 0], name="This is name for part of z1", print_shape=True, print_ptr=True)
    z2 = z1 @ y
    debug_print.print_tensor(z2, name="This is name for z2")

debug_print.post_initialize()

x.copy_(torch.randn(2, 2))
y.copy_(torch.ones(2, 2))
print("start replay...")
g.replay()
