[code]
!git clone https://github.com/open-compass/VLMEvalKit.git
%cd VLMEvalKit
!pip install -e .

[code]
# Uninstall the problematic version
!pip uninstall transformers -y

# Then install a recent version that includes AutoModelForImageTextToText
!pip install transformers==4.47.0

# Verify installation
!python -c "import transformers; print(transformers.__version__)"
!pip install flash-attn --no-build-isolation

[code]
%cd VLMEvalKit

[code]
# Verify the version first
import transformers
print(transformers.__version__) # Should be 4.47.0 or higher

# Then run your evaluation
!python run.py --data DynaMath --model Qwen2-VL-7B-Instruct --judge Gemini-1.5-Pro --verbose

[code]
!git clone https://github.com/Vasudev08/reasonvlm.git

[code]
%cd reasonvlm
!git pull

[code]
# %cd reasonvlm
!python setup_colab.py

[code]
import os
os.environ['OPENAI_API_KEY'] = ''

[code]
!pip install vllm

[code]
from google.colab import drive
import os
drive.mount('/content/drive')

# Create the output folder and copy your backup back in
!mkdir -p /content/reasonvlm/VLMEvalKit/outputs/Qwen2-VL-7B-Instruct
!cp -r /content/drive/MyDrive/VL_Backup_Safe/* /content/reasonvlm/VLMEvalKit/outputs/Qwen2-VL-7B-Instruct/

[code]
!python VLMEvalKit/run.py --data DynaMath --model Qwen2-VL-7B-Instruct --verbose --judge gpt-4o-mini --mode eval --reuse --use-vllm

[code]
