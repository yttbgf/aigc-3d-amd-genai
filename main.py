import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import rembg
import common.global_constant
from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize


# import nvdiffrast.torch as dr

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui  # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.fixed_cam = None
        self.fixed_cam_list = []
        self.mode = "image"
        self.seed = "random"

        # only for gui
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device(common.global_constant.GPU.DEVICE)
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussian_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        # {{ only for GUI
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5
        # }}
        self.input_img_list = []
        self.input_mask_list = []
        self.input_img_torch_list = []
        self.input_mask_torch_list = []

        # input text
        self.prompt = ""
        self.negative_prompt = ""
        self.prompt_list = []
        self.negative_prompt_list = []

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)

        if self.opt.input_files is not None:
            input_files = self.opt.input_files.split(',')
            self.load_input_files(input_files)
        else:
            if self.input_img is not None:
                self.input_img_list.append(self.input_img)
                self.input_mask_list.append(self.input_mask)
        # print(self.input_img, self.input_mask)
        # print(self.input_img_list, self.input_mask_list, self.prompt_list)

        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt
        print('prompt:', self.prompt, '; negative_prompt:', self.negative_prompt)
        # exit(0)

        # override if provide a checkpoint
        print('load:', self.opt.load)
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self, is_load_guidance=True):
        print('prepare_train ... is_load_guidance:', is_load_guidance)
        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )
        # TODO: self.fixed_cam_list
        if self.opt.input_camera_pose is not None:
            input_camera_poses = self.opt.input_camera_pose.split(',')
            for pose in input_camera_poses:
                pose = float(pose)
                print('pose: %s' % pose)
                if self.opt.mvdream or self.opt.imagedream:
                    # the second view is the front view for mvdream/imagedream.
                    pose = orbit_camera(self.opt.elevation, 90 + pose, self.opt.radius)
                else:
                    pose = orbit_camera(self.opt.elevation, pose, self.opt.radius)
                self.fixed_cam_list.append(MiniCam(
                    pose,
                    self.opt.ref_size,
                    self.opt.ref_size,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                ))
        else:
            self.fixed_cam_list.append(self.fixed_cam)

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        print('lambda_sd:', self.opt.lambda_sd)
        print('enable_sd:', self.enable_sd)

        self.enable_zero123 = self.opt.lambda_zero123 > 0 and (
                self.input_img is not None or len(self.input_img_list) > 0)
        print('lambda_zero123:', self.opt.lambda_zero123)
        print('enable_zero123:', self.enable_zero123)
        # lazy load guidance model
        print('guidance_sd:', self.guidance_sd)

        if not is_load_guidance:
            return

        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        print('guidance_zero123:', self.guidance_zero123)
        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            model_key = None
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                model_key = 'ashawkey/stable-zero123-diffusers'
                self.guidance_zero123 = Zero123(self.device, model_key=model_key)
            else:
                model_key = 'ashawkey/zero123-xl-diffusers'
                self.guidance_zero123 = Zero123(self.device, model_key=model_key)
            print(f"[INFO] loaded zero123!", model_key)

        # input image
        '''
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size),
                                                 mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size),
                                                  mode="bilinear", align_corners=False)
        '''
        # TODO: self.input_img_torch_list  input_mask_torch_list
        print('input_img_list:', len(self.input_img_list))
        for img_idx in range(len(self.input_img_list)):
            input_img_torch = torch.from_numpy(self.input_img_list[img_idx]).permute(2, 0, 1).unsqueeze(0).to(self.device)
            input_img_torch = F.interpolate(input_img_torch, (self.opt.ref_size, self.opt.ref_size),
                                            mode="bilinear", align_corners=False)
            self.input_img_torch_list.append(input_img_torch)

            input_mask_torch = torch.from_numpy(self.input_mask_list[img_idx]).permute(2, 0, 1).unsqueeze(0).to(self.device)
            input_mask_torch = F.interpolate(input_mask_torch, (self.opt.ref_size, self.opt.ref_size),
                                             mode="bilinear", align_corners=False)
            self.input_mask_torch_list.append(input_mask_torch)

        # prepare embeddings
        with torch.no_grad():
            # TODO: self.input_img_torch_list  just use first image calc embedding because utils train_step not handle multi
            # prompts negative_prompts support multi?but encode_text in utils not support?
            # support multi image embedding in future? necessary?
            input_img_torch = self.input_img_torch_list[0] if (len(self.input_img_torch_list) > 0) else None
            if self.enable_sd:
                if self.opt.imagedream:  # NOTICE: mvdream only use text input, imagedream both?
                    self.guidance_sd.get_image_text_embeds(input_img_torch, [self.prompt], [self.negative_prompt])
                else:  # only use text input
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:  # only use image input
                self.guidance_zero123.get_img_embeds(input_img_torch)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for step in range(self.train_steps):  # 1??
            print('step:', step)
            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### known view has ground truth
            if not self.opt.imagedream:  # NOTICE: imagedream not use as ground truth?
                print('known view ground truth loss...')
                for idx_image in range(len(self.input_img_torch_list)):
                    # TODO: set cur_cam
                    #cur_cam = self.fixed_cam
                    cur_cam = self.fixed_cam_list[idx_image]
                    out = self.renderer.render(cur_cam)

                    # rgb loss
                    image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                    mse_loss_rgb = F.mse_loss(image, self.input_img_torch_list[idx_image])
                    print('mse_loss_rgb:', mse_loss_rgb)
                    lambda_loss_rgb = 10000
                    loss = loss + lambda_loss_rgb * (step_ratio if self.opt.warmup_rgb_loss else 1) * mse_loss_rgb

                    # mask loss
                    mask = out["alpha"].unsqueeze(0)  # [1, 1, H, W] in [0, 1]
                    mse_loss_alpha = F.mse_loss(mask, self.input_mask_torch_list[idx_image])
                    print('mse_loss_alpha:', mse_loss_alpha)
                    lambda_loss_alpha = 1000
                    loss = loss + lambda_loss_alpha * (step_ratio if self.opt.warmup_rgb_loss else 1) * mse_loss_alpha

            ### novel view (manual batch) no ground truth so use guidance loss
            print('novel view guidance loss...')
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)
            print('batch_size:', self.opt.batch_size)
            for batch_i in range(self.opt.batch_size):
                print('batch_i:', batch_i)
                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                # print('pose:', pose.shape)
                # print('pose after orbit_camera', pose)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx,
                                  self.cam.near, self.cam.far)
                # print('pose after MiniCam', pose)
                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0],
                                        dtype=torch.float32, device=common.global_constant.GPU.DEVICE)
                print('render ...', batch_i)
                out = self.renderer.render(cur_cam, bg_color=bg_color)
                print('render finish!', batch_i)
                # print('pose after render', pose)
                # print(out.keys())
                # print('image:', out["image"].shape)
                # print('image:', out["image"]) #crash!
                image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                print('image:', image.shape)
                # print('image:', image)
                images.append(image)  # current random view

                # enable mvdream training
                if self.opt.mvdream or self.opt.imagedream:  # other views: rotate horizontally based random view
                    print('mvdream or imagedream...')
                    for view_i in range(1, 4):  # 3 views each rotate 90deg
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                        print('pose_i:', pose_i.shape)
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx,
                                            self.cam.near, self.cam.far)

                        # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=common.global_constant.GPU.DEVICE)
                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)
                        # print('out_i:', out_i)
                        image = out_i["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                        print('image:', image.shape)
                        images.append(image)
                    print('mvdream or imagedream finish!')
            print('batch all finish!', len(images))
            images = torch.cat(images, dim=0)
            print('images finish!', images.shape, 'poses:', len(poses))
            # print('images:', images)
            # poses1 = torch.ones(1, 4, 4)

            # print('pose1 after from_numpy', poses1, poses1.dtype)

            # poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
            poses = np.stack(poses, axis=0)
            # print('pose after stack', poses, poses.dtype) #crash!!
            # poses = torch.from_numpy(np.ascontiguousarray(poses))
            # poses = torch.from_numpy(poses.copy())
            poses = torch.from_numpy(poses)
            # print('pose shape after from_numpy', poses.shape)
            # poses = np.ones_like(poses)
            # print('pose after from_numpy', poses)
            # poses = torch.from_numpy(poses)
            # poses = torch.ones_like(poses)
            # poses = torch.ones(1, 4, 4)

            # print('pose after from_numpy', poses, poses.dtype)

            # print('pose1 after from_numpy', poses1, poses1.dtype)

            # poses2 = torch.ones(1, 4, 4)

            # print('pose2 after from_numpy', poses2, poses2.dtype)
            # print('device:', self.device)
            # poses2 = poses2.to(self.device)
            # print('pose2 after to', poses2)
            # print('poses to:', self.device)
            poses = poses.to(self.device)
            # print('pose after to', poses)

            # import kiui
            # print(hor, ver)
            # kiui.vis.plot_image(images)

            # guidance loss
            print('guidance loss start')
            # TODO:
            #  all has embeddings which inited in prepare_train should handle multi images or texts
            #  mvdream:text imagedream image&text sd:text zero123:image
            #  only zero123 has cam_embeddings for novel views
            if self.enable_sd:
                print('guidance_sd.train_step...')
                if self.opt.mvdream or self.opt.imagedream:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses,
                                                                                   step_ratio=step_ratio if self.opt.anneal_timestep else None)
                else:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images,
                                                                                   step_ratio=step_ratio if self.opt.anneal_timestep else None)
                print('guidance_sd.train_step finish!')
            if self.enable_zero123:
                print('guidance_zero123.train_step ...')
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii,
                                                                                         step_ratio=step_ratio if self.opt.anneal_timestep else None,
                                                                                         default_elevation=self.opt.elevation)
                print('guidance_zero123.train_step finish!')
            # optimize step
            print('optimize step ...')
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print('optimize step finish!')

            # densify and prune
            # No training functions
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                print('densify and prune')
                #print(out)
                #print(out["visibility_filter"])
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], \
                    out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01,
                                                              extent=4, max_screen_size=1)

                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()
        print('record...')
        ender.record()
        print('record finish!')
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussian_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (
                            buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                        self.buffer_image * (1 - self.overlay_input_img_ratio)
                        + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000 / t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    def load_input(self, file):
        print('file:', file)
        self.input_img, self.input_mask = self.load_image(file)
        # print(self.input_img, self.input_mask)
        # exit(0)
        self.prompt = self.load_prompt(file)
        print('prompt：', self.prompt)

    def load_input_files(self, files):
        for file in files:
            print('file:', file)
            input_img, input_mask = self.load_image(file)
            self.input_img_list.append(input_img)
            self.input_mask_list.append(input_mask)
            prompt = self.load_prompt(file)
            print('prompt：', prompt)
            self.prompt_list.append(prompt)

    def load_image(self, file):
        # load image
        print(f'[INFO] load image from {file} ...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        print('img shape:', img.shape)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        input_mask = img[..., 3:]
        # white bg
        input_img = img[..., :3] * input_mask + (1 - input_mask)
        # bgr to rgb
        input_img = input_img[..., ::-1].copy()
        return input_img, input_mask

    def load_prompt(self, file):
        # load prompt
        prompt = ""
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                prompt = f.read().strip()
        return prompt

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            print(f"[INFO] geo extract_mesh ...")
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            print('write_ply...')
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            print(f"[INFO] geo+tex extract_mesh ...")
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()
            print("mesh:", mesh)
            # TODO: dump everything use another python file call nvidia!
            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512
            print('loading nvdiffrast...')
            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                print('RasterizeGLContext')
                glctx = dr.RasterizeGLContext()
            else:
                print('RasterizeCudaContext')
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                print('ver, hor:', ver, hor)
                # render image
                print('orbit_camera...')
                pose = orbit_camera(ver, hor, self.cam.radius)
                print('MiniCam...')
                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                print('render cur_cam...')
                cur_out = self.renderer.render(cur_cam)
                print('render cur_cam finish!')
                rgbs = cur_out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                # import kiui
                # kiui.vis.plot_image(rgbs)

                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)
                print('v_cam matmul...')
                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0),
                                     torch.inverse(pose).T).float().unsqueeze(0)
                print('v_clip matmul...')
                v_clip = v_cam @ proj.T
                print('dr rasterize...')
                # only this use subprocess call nvidia!! notice glctx not in the loop!
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))
                print('dr interpolate 1...')
                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f)  # [1, H, W, 1]
                depth = depth.squeeze(0)  # [H, W, 1] #Notice: return value not used???

                alpha = (rast[0, ..., 3:] > 0).float()
                print('dr interpolate 2...')
                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                print('dr interpolate 3...')
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                print('rot_normal...')
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()

                # update texture image
                print('mipmap_linear_grid_put_2d...')
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )

                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion
            print('binary_dilation...')
            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            print('binary_erosion...')
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            print('kneighbors...')
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            print('save_ply...')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path} .")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
                tag="_primary_window",
                width=self.W,
                height=self.H,
                pos=[0, 0],
                no_move=True,
                no_title_bar=True,
                no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
                label="Control",
                tag="_control_window",
                width=600,
                height=self.H,
                pos=[self.W, 0],
                no_move=True,
                no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                        directory_selector=False,
                        show=False,
                        callback=callback_select_input,
                        file_count=1,
                        tag="file_dialog_tag",
                        width=700,
                        height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")

                # overlay stuff
                with dpg.group(horizontal=True):
                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff

                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussian_scale(sender, app_data):
                    self.gaussian_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussian scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussian_scale_factor,
                    callback=callback_set_gaussian_scale,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                print('train_step ...')
                self.train_step()
                print('train_step finish!')
            print('test_step ...')
            self.test_step()
            print('test_step finish!')
            dpg.render_dearpygui_frame()
            print('render_dearpygui_frame finish!')

    # no gui mode
    def train(self, iters=500, is_amd=False):
        if iters > 0:
            print('prepare_train ...')
            self.prepare_train()
            print('prepare_train finish!')
            for i in tqdm.trange(iters):
                self.train_step()
            print('train_steps finish!iters:', iters)
            # do a last prune
            print('gaussians prune ...')
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
            print('gaussians prune finish!')
        # save
        print('save_model model ...')
        self.save_model(mode='model')
        print('save_model model finish!')
        path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh_before.pcd')
        self.renderer.gaussians.save_pcd(path)
        if not is_amd:
            print('save_model geo+tex ...')
            self.save_model(mode='geo+tex')
            print('save_model geo+tex finish!')
        else:
            self.dump_render()
        path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh_after.pcd')
        self.renderer.gaussians.save_pcd(path)

    def resume_train(self, iters=0, is_amd=False):
        if is_amd:
            self.load_render()

        self.prepare_train(is_load_guidance=False)
        print('prepare_train finish!')
        self.save_model(mode='geo+tex')
        print('save_model geo+tex finish!')
        path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh_after_resume.pcd')
        self.renderer.gaussians.save_pcd(path)

    def dump_render(self):
        pass

    def load_render(self):
        pass


if __name__ == "__main__":
    '''
    import nvdiffrast.torch as dr

    print('RasterizeGLContext')
    glctx = dr.RasterizeGLContext()
    #glctx = dr.RasterizeCudaContext() #only for nivida?or cuda?
    exit(0)
    '''
    #torch.backends.cuda.preferred_linalg_library('default')
    #torch.backends.cuda.preferred_linalg_library('cusolver')
    torch.backends.cuda.preferred_linalg_library('magma')
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    # print('args:', args, 'extras:', extras)
    # exit(0)
    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.gui:
        print('gui render')
        gui.render()
    else:
        print('gui train')
        if opt.resume is not None and opt.resume:
            #print('is_amd:', opt.is_amd)
            gui.resume_train(opt.iters)
            #gui.resume_train(opt.iters, is_amd=opt.is_amd)
        else:
            #exit(0)
            is_amd = False
            if opt.is_amd is not None:
                is_amd = opt.is_amd
            print('is_amd:', is_amd)
            gui.train(opt.iters, is_amd=is_amd)
