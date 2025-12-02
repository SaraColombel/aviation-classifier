import torch
import torchvision
import matplotlib
import streamlit

print("PyTorch version :", torch.__version__)
print("CUDA dispo ?     :", torch.cuda.is_available())
print("Torchvision OK   :", torchvision.__version__)
print("Matplotlib OK    :", matplotlib.__version__)
print("Streamlit OK     :", streamlit.__version__)
