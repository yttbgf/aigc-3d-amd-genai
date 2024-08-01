# aigc-3d-amd-genai

This repository contains the implementation for AMD Generative AI Contest 2003 [3D creative food modeling generation by text and picture](https://news.elecnest.cn/contests_amd2023).

## Install

```bash
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
pip install ./diff-gaussian-rasterization

# a modified simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit

# To use MVdream, also install:
pip install git+https://github.com/bytedance/MVDream

```

Tested on:

- Ubuntu 22 with torch 2.3 & rocm 6.0.2-6.1.3 on a AMD Radeon™ PRO W7900.

## Usage

Image-to-3D:

```bash
### preprocess
# background removal and recentering, save rgba at 256x256
python process.py data/name.jpg

# save at a larger resolution
python process.py data/name.jpg --size 512

# process all jpg images under a dir
python process.py data

### training gaussian stage
# train 500 iters (~1min) and export ckpt & coarse_mesh to logs
python main.py --config configs/image.yaml input=data/name_rgba.png save_path=name

# load and visualize a saved ckpt
python main.py --config configs/image.yaml load=logs/name_model.ply gui=True

# use an estimated elevation angle if image is not front-view (e.g., common looking-down image can use -30)
python main.py --config configs/image.yaml input=data/name_rgba.png save_path=name elevation=-30

### training mesh stage
# auto load coarse_mesh and refine 50 iters (~1min), export fine_mesh to logs
python main2.py --config configs/image.yaml input=data/name_rgba.png save_path=name

# specify coarse mesh path explicity
python main2.py --config configs/image.yaml input=data/name_rgba.png save_path=name mesh=logs/name_mesh.obj

### training gaussian stage for multi-view images input
python main.py --config configs/image.yaml input_files=aigc/object/crane_machine/input/IMG_20240730_1_rgba.png,aigc/object/crane_machine/input/IMG_20240730_2_rgba.png,aigc/object/crane_machine/input/IMG_20240730_3_rgba.png,aigc/object/crane_machine/input/IMG_20240730_4_rgba.png,aigc/object/crane_machine/input/IMG_20240730_5_rgba.png,aigc/object/crane_machine/input/IMG_20240730_6_rgba.png,aigc/object/crane_machine/input/IMG_20240730_7_rgba.png,aigc/object/crane_machine/input/IMG_20240730_8_rgba.png input_camera_pose=0,45,90,135,180,225,270,315 save_path=aigc/object/crane_machine/output_multi_img_amd

### training mesh stage for multi-view images input
python main2.py --config configs/image.yaml input_files=aigc/object/crane_machine/input/IMG_20240730_1_rgba.png,aigc/object/crane_machine/input/IMG_20240730_2_rgba.png,aigc/object/crane_machine/input/IMG_20240730_3_rgba.png,aigc/object/crane_machine/input/IMG_20240730_4_rgba.png,aigc/object/crane_machine/input/IMG_20240730_5_rgba.png,aigc/object/crane_machine/input/IMG_20240730_6_rgba.png,aigc/object/crane_machine/input/IMG_20240730_7_rgba.png,aigc/object/crane_machine/input/IMG_20240730_8_rgba.png input_camera_pose=0,45,90,135,180,225,270,315 save_path=aigc/object/crane_machine/output_multi_img_amd

### visualization
# gui for visualizing mesh
# `kire` is short for `python -m kiui.render`
kire logs/name.obj

# save 360 degree video of mesh (can run without gui)
kire logs/name.obj --save_video name.mp4 --wogui

# save 8 view images of mesh (can run without gui)
kire logs/name.obj --save images/name/ --wogui

```

Please check `./configs/image.yaml` for more options.


Text-to-3D:

```bash
### training gaussian stage
python main.py --config configs/text.yaml prompt="a photo of an cookie whose pattern is of Mickey Mouse in a tuxedo and Minnie Mouse in a blue swan lake ballet costume dancing together" save_path=cookie

### training mesh stage
python main2.py --config configs/text.yaml prompt="a photo of an cookie whose pattern is of Mickey Mouse in a tuxedo and Minnie Mouse in a blue swan lake ballet costume dancing together" save_path=cookie
```

Please check `./configs/text.yaml` for more options.

Text-to-3D (MVDream):

```bash
### training gaussian stage
python main.py --config configs/text_mv.yaml prompt="heart-shaped ice cream cones,connected by a stick,with matcha and blueberry flavors sprinkled with chocolate chips or sauce" save_path=heart_icecream

### training mesh stage
python main2.py --config configs/text_mv.yaml prompt="heart-shaped ice cream cones,connected by a stick,with matcha and blueberry flavors sprinkled with chocolate chips or sauce" save_path=heart_icecream
```

Please check `./configs/text_mv.yaml` for more options.


Helper scripts:

```bash

# export all ./logs/*.obj to mp4 in ./videos
python scripts/convert_obj_to_video.py --dir ./logs
```


## Tips
* The world & camera coordinate system is the same as OpenGL:
```
    World            Camera        
  
     +y              up  target                                              
     |               |  /                                            
     |               | /                                                
     |______+x       |/______right                                      
    /                /         
   /                /          
  /                /           
 +z               forward           

elevation: in (-90, 90), from +y to -y is (-90, 90)
azimuth: in (-180, 180), from +z to +x is (0, 90)
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)
