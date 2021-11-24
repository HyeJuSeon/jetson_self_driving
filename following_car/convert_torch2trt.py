from torchvision.models import resnet18
import torch
from torch.nn import Linear
from torch2trt import torch2trt

model = resnet18(pretrained=False)
model.fc = Linear(512, 2)
model.load_state_dict(torch.load('best_resnet.pth'))

device = torch.device('cuda')
model = model.to(device)
model = model.cuda().eval().half()

data = torch.randn((1, 3, 224, 224)).cuda().half()
model_trt = torch2trt(model, [data], fp16_mode=True)

output_trt = model_trt(data)
output = model(data)

print(output.flatten()[0:10])
print(output_trt.flatten()[0:10])
print(f'max error: %{float(torch.max(torch.abs(output - output_trt)))}')
torch.save(model_trt.state_dict(), 'best_resnet_trt.pth')