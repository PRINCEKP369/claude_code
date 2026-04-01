# claude_code

npm install -g ./antxxxxx

# add env vars

``` bash
export ANTHROPIC_BASE_URL="http://localhost:8080"   # no /v1 suffix
export ANTHROPIC_AUTH_TOKEN="local-llama-cpp"
export ANTHROPIC_API_KEY=""
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

```

Sure. The idea is simple: MATLAB writes the input to a file, calls a Python script, Python runs BeamNet and writes the output, MATLAB reads it back.

Here's everything you need:

---

## Step 1 — Python inference script (`infer.py`)

Save this next to your `beamnet_best.pth`:

```python
import torch
import scipy.io as sio
import numpy as np
import argparse
from Model_v3 import BeamNet   # your model file

parser = argparse.ArgumentParser()
parser.add_argument("--input",  required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# Load input from MATLAB
data    = sio.loadmat(args.input)
x       = np.float32(data["input_vec"])   # expects [1 x 180]

# Load model
checkpoint = torch.load("beamnet_best.pth", map_location="cpu")
model = BeamNet(base_ch=64)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Normalize (same stats used in training)
mean = float(checkpoint["input_mean"])
std  = float(checkpoint["input_std"])
x_norm = (x - mean) / (std + 1e-8)

# Run inference
with torch.no_grad():
    x_tensor = torch.tensor(x_norm).reshape(1, 1, 180)
    y = model(x_tensor).squeeze().numpy()   # [180,]

# Save output for MATLAB
sio.savemat(args.output, {"output_vec": y})
print("done")
```

---

## Step 2 — MATLAB side

```matlab
% --- Your MVDR output (1x180 vector) ---
input_vec = single(mvdr_power);          % make sure it's [1 x 180]

% Save to temp file
save("temp_input.mat", "input_vec");

% Call Python
status = system("python3 infer.py --input temp_input.mat --output temp_output.mat");

if status ~= 0
    error("Python inference failed. Check infer.py and paths.");
end

% Load result
result        = load("temp_output.mat");
output_vec    = result.output_vec;       % [1 x 180], values in (0,1)

% Find predicted target angles
threshold     = 0.5;
target_angles = find(output_vec > threshold);  % degree bins where peak detected
disp(target_angles);
```

---

## Folder structure

Everything should be in the same folder to keep paths simple:

```
your_project/
├── infer.py
├── Model_v3.py
├── beamnet_best.pth
├── temp_input.mat       ← created by MATLAB at runtime
├── temp_output.mat      ← created by Python at runtime
└── your_matlab_script.m
```

---

## Two things to check if it fails

**1. Python path** — if `python3` isn't found, find the full path first:
```bash
which python3        # Linux
```
Then replace `"python3"` in the `system()` call with the full path:
```matlab
system("/usr/bin/python3 infer.py --input temp_input.mat --output temp_output.mat");
```

**2. Working directory** — MATLAB's working directory must match where `infer.py` lives. Either `cd` to that folder in MATLAB, or use absolute paths:
```matlab
system("/home/prince/project/infer.py ...")
% or add cd at the top of your script:
cd("/home/prince/project");
```

That's the whole setup — roughly 30 lines total between both files.
