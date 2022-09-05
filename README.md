# diffusion-gen
Use stable diffusion to generate images guided by text

# INSTALLATION

git clone https://github.com/CompVis/stable-diffusion
cd stable-diffusion
conda env create -f environment.yaml
conda activate ldm

conda install -c conda-forge diffusers

# USAGE

python main.py [filename.jpg] --prompt "prompt text here"

NOTE: do not change the order of the arguments

You can choose to skip the prompt, in which case it will just use "a high resolution photograph"



