import torch
print(torch.__version__)  # Should show the nightly version.
print(torch.cuda.is_available())  # Should return True if CUDA is properly set up.
print(torch.version.cuda)  # Should match CUDA 12.6.
