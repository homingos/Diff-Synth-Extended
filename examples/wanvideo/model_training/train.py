import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import (
    DiffusionTrainingModule,
    ModelLogger,
    launch_training_task,
    wan_parser,
)
from diffsynth.trainers.unified_dataset import (
    UnifiedDataset,
    LoadVideo,
    LoadAudio,
    ImageCropAndResize,
    ToAbsolutePath,
    LoadRGBAVideoPair,
    HardRenderRGBA,
    SoftRenderRGBA,
)
from diffsynth.models.wan_video_vae import RGBAlphaVAE

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths, enable_fp8_training=False
        )
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(
                model_id=audio_processor_config.split(":")[0],
                origin_file_pattern=audio_processor_config.split(":")[1],
            )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            audio_processor_config=audio_processor_config,
        )

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif (
                extra_input == "reference_image"
                or extra_input == "vace_reference_image"
            ):
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {
            name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models
        }
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


class WanAlphaTrainingModule(WanTrainingModule):
    """
    Extended WanTrainingModule for RGBA DiT training with DoRA (Wan-Alpha Stage 2).
    Uses trained FeatureMergeBlock to encode RGBA videos.
    """

    def __init__(
        self,
        rgba_mode=False,
        base_vae_path=None,
        vae_lora_path=None,
        feature_merge_checkpoint=None,
        use_dora=False,
        dora_rank=32,
        dora_checkpoint=None,
        **kwargs,
    ):
        """
        Args:
            rgba_mode: Enable RGBA training mode
            base_vae_path: Path to base Wan2.1_VAE.pth
            vae_lora_path: Path to decoder.bin
            feature_merge_checkpoint: Path to trained FeatureMergeBlock weights
            use_dora: Use DoRA instead of LoRA for DiT
            dora_rank: DoRA rank
            dora_checkpoint: Path to DoRA checkpoint
            **kwargs: Other WanTrainingModule arguments
        """
        # Initialize base module
        super().__init__(**kwargs)

        self.rgba_mode = rgba_mode

        if rgba_mode:
            # Setup RGBA VAE with trained FeatureMergeBlock
            self.rgba_vae = RGBAlphaVAE(
                base_vae_path=base_vae_path,
                vae_lora_path=vae_lora_path,
                z_dim=16,
                dtype=torch.bfloat16,
                device="cpu",
                with_feature_merge=True,
            )

            # Load trained FeatureMergeBlock weights
            if feature_merge_checkpoint is not None:
                print(
                    f"Loading trained FeatureMergeBlock from {feature_merge_checkpoint}"
                )
                merge_state = load_state_dict(feature_merge_checkpoint)
                self.rgba_vae.feature_merge.load_state_dict(merge_state)
                print("✅ FeatureMergeBlock loaded successfully")

            # Freeze everything except DiT
            self.rgba_vae.freeze_decoders()
            self.rgba_vae.freeze_feature_merge()

            # Switch to DoRA training if enabled
            if use_dora:
                self.switch_pipe_to_training_mode(
                    self.pipe,
                    kwargs.get("trainable_models"),
                    kwargs.get("lora_base_model"),
                    kwargs.get("lora_target_modules"),
                    kwargs.get("lora_rank"),
                    lora_checkpoint=kwargs.get("lora_checkpoint"),
                    enable_fp8_training=False,
                    use_dora=True,
                    dora_rank=dora_rank,
                    dora_checkpoint=dora_checkpoint,
                )

    def forward_preprocess(self, data):
        """Override to handle RGBA video encoding"""
        if self.rgba_mode:
            # Extract RGBA components from data
            rgb_video = data.get("rgb_video")
            alpha_video = data.get("alpha_video")
            hard_rgb_video = data.get("hard_rgb_video")

            # Convert PIL images to tensors for VAE encoding
            # Use hard-rendered RGB for encoding (prevents color/transparency confusion)
            hard_rgb_tensors = [self.pil_list_to_tensor(hard_rgb_video)]
            alpha_tensors = [self.pil_list_to_tensor(alpha_video)]

            # Encode with trained FeatureMergeBlock
            with torch.no_grad():
                merged_latents = self.rgba_vae.encode_with_merge(
                    hard_rgb_tensors, alpha_tensors, tiled=False
                )

            # Create input dict for DiT training
            inputs_posi = {"prompt": data["prompt"]}
            inputs_nega = {}

            inputs_shared = {
                "input_video": merged_latents[0],  # Use merged latent as "video"
                "height": data["rgb_video"][0].size[1],
                "width": data["rgb_video"][0].size[0],
                "num_frames": len(data["rgb_video"]),
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
            }

            # Process through pipeline units
            for unit in self.pipe.units:
                inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                    unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
                )

            return {**inputs_shared, **inputs_posi}
        else:
            # Standard video training
            return super().forward_preprocess(data)

    def pil_list_to_tensor(self, pil_images):
        """Convert list of PIL images to tensor [C, T, H, W]"""
        import torchvision.transforms as transforms

        to_tensor = transforms.ToTensor()
        tensors = [to_tensor(img) for img in pil_images]
        video_tensor = torch.stack(tensors, dim=1)  # [C, T, H, W]
        video_tensor = video_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        return video_tensor


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    # Setup dataset based on mode
    if args.rgba_mode:
        # RGBA dataset with paired RGB+alpha videos
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=ToAbsolutePath(args.dataset_base_path)
            >> LoadRGBAVideoPair(
                num_frames=args.num_frames,
                frame_processor=ImageCropAndResize(
                    args.height, args.width, None, 16, 16
                ),
            )
            >> HardRenderRGBA(),
        )
    else:
        # Standard video dataset
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path)
                >> LoadVideo(
                    args.num_frames,
                    4,
                    1,
                    frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
                ),
                "input_audio": ToAbsolutePath(args.dataset_base_path)
                >> LoadAudio(sr=16000),
            },
        )

    # Setup model based on mode
    if args.rgba_mode:
        model = WanAlphaTrainingModule(
            rgba_mode=True,
            base_vae_path=args.base_vae_path,
            vae_lora_path=args.vae_lora_path,
            feature_merge_checkpoint=args.feature_merge_checkpoint,
            use_dora=args.use_dora,
            dora_rank=args.dora_rank,
            dora_checkpoint=args.dora_checkpoint,
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=args.lora_checkpoint,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            extra_inputs=args.extra_inputs,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
        )
    else:
        model = WanTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            audio_processor_config=args.audio_processor_config,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=args.lora_checkpoint,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            extra_inputs=args.extra_inputs,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
        )
    model_logger = ModelLogger(
        args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)
