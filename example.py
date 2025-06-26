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
    debug_print.print_tensor(z1[..., 0], name="This is name for part of z1")
    z2 = z1 @ y
    debug_print.print_tensor(z2, name="This is name for z2")

debug_print.post_initialize()

x.copy_(torch.randn(2, 2))
y.copy_(torch.ones(2, 2))
print("start replay...")
g.replay()
