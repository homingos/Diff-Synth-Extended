import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = (
            [] if operators is None else operators
        )

    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value

    def __call__(self, data):
        if data is None:
            data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB

    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB:
            image = image.convert("RGB")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(
        self, height, width, max_pixels, height_division_factor, width_division_factor
    ):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = torchvision.transforms.functional.center_crop(
            image, (target_height, target_width)
        )
        return image

    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]


class LoadVideo(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while (
                num_frames > 1
                and num_frames % self.time_division_factor
                != self.time_division_remainder
            ):
                num_frames -= 1
        return num_frames

    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator

    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while (
                num_frames > 1
                and num_frames % self.time_division_factor
                != self.time_division_remainder
            ):
                num_frames -= 1
        return num_frames

    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location

    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        # Convert to string if needed (handles both str and int from CSV)
        if isinstance(data, int):
            # If integer, pad with zeros to match file naming (e.g., 1 → "001")
            data_str = f"{data:03d}"
        elif isinstance(data, str):
            data_str = data
        else:
            data_str = str(data)
        return os.path.join(self.base_path, data_str)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr

    def __call__(self, data: str):
        import librosa

        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class SoftRenderRGBA(DataProcessingOperator):
    """
    Apply soft rendering to RGBA video as per Wan-Alpha paper (Equation 1).
    Soft rendering: R_s = V_rgb * α + c * (1 - α), where α ∈ [0, 1]

    Used for training loss calculation (L_rgb^s).
    """

    def __init__(self, color_set=None):
        """
        Args:
            color_set: List of RGB tuples for random background colors
        """
        self.color_set = color_set or [
            (0, 0, 0),  # black
            (0, 0, 255),  # blue
            (0, 255, 0),  # green
            (0, 255, 255),  # cyan
            (255, 0, 0),  # red
            (255, 0, 255),  # magenta
            (255, 255, 0),  # yellow
            (255, 255, 255),  # white
        ]

    def __call__(self, data: dict):
        """
        Apply soft rendering to RGB video using alpha video.

        Args:
            data: Dict with 'rgb_video' and 'alpha_video' keys

        Returns:
            dict: Original dict plus 'soft_rgb_video' and 'soft_color'
        """
        import random
        import numpy as np

        rgb_frames = data["rgb_video"]
        alpha_frames = data["alpha_video"]

        # Randomly select background color
        soft_color = random.choice(self.color_set)

        # Apply soft rendering to each frame
        soft_rgb_frames = []
        for rgb_img, alpha_img in zip(rgb_frames, alpha_frames):
            # Convert to numpy arrays
            rgb_array = np.array(rgb_img).astype(np.float32)

            # Convert alpha to [0, 1] range (soft alpha, continuous values)
            if alpha_img.mode == "L":
                alpha_array = np.array(alpha_img).astype(np.float32) / 255.0
            else:
                # If RGB format (3-channel), take mean
                alpha_array = (
                    np.array(alpha_img).astype(np.float32).mean(axis=2) / 255.0
                )

            # Expand alpha to match RGB dimensions
            if len(alpha_array.shape) == 2:
                alpha_array = np.expand_dims(alpha_array, axis=2)

            # Apply soft rendering: RGB * alpha + color * (1 - alpha)
            soft_rgb = rgb_array * alpha_array + np.array(soft_color) * (
                1 - alpha_array
            )
            soft_rgb = np.clip(soft_rgb, 0, 255).astype(np.uint8)

            # Convert back to PIL Image
            soft_rgb_img = Image.fromarray(soft_rgb)
            soft_rgb_frames.append(soft_rgb_img)

        # Add soft-rendered video to data dict
        data["soft_rgb_video"] = soft_rgb_frames
        data["soft_color"] = soft_color

        return data


class HardRenderRGBA(DataProcessingOperator):
    """
    Apply hard rendering to RGBA video as per Wan-Alpha paper (Equation 2).
    Hard rendering: R_h = V_rgb * α + c * (1 - α), where α = 1 if α > 0, else 0

    This prevents the encoder from confusing RGB background color with transparency.
    """

    def __init__(self, color_set=None, alpha_threshold=0.5):
        """
        Args:
            color_set: List of RGB tuples for random background colors
            alpha_threshold: Threshold to binarize alpha channel (default 0.5)
        """
        self.color_set = color_set or [
            (0, 0, 0),  # black
            (0, 0, 255),  # blue
            (0, 255, 0),  # green
            (0, 255, 255),  # cyan
            (255, 0, 0),  # red
            (255, 0, 255),  # magenta
            (255, 255, 0),  # yellow
            (255, 255, 255),  # white
        ]
        self.alpha_threshold = alpha_threshold

    def __call__(self, data: dict):
        """
        Apply hard rendering to RGB video using alpha video.

        Args:
            data: Dict with 'rgb_video' and 'alpha_video' keys

        Returns:
            dict: Original dict plus 'hard_rgb_video' and 'hard_color'
        """
        import random
        import numpy as np

        rgb_frames = data["rgb_video"]
        alpha_frames = data["alpha_video"]

        # Randomly select background color
        hard_color = random.choice(self.color_set)

        # Apply hard rendering to each frame
        hard_rgb_frames = []
        for rgb_img, alpha_img in zip(rgb_frames, alpha_frames):
            # Convert to numpy arrays
            rgb_array = np.array(rgb_img).astype(np.float32)
            alpha_array = np.array(alpha_img).astype(np.float32) / 255.0

            # Create hard alpha mask (binarize)
            hard_alpha = (alpha_array > self.alpha_threshold).astype(np.float32)

            # Expand alpha to match RGB dimensions
            if len(hard_alpha.shape) == 2:
                hard_alpha = np.expand_dims(hard_alpha, axis=2)

            # Apply hard rendering: RGB where alpha > threshold, color elsewhere
            hard_rgb = rgb_array * hard_alpha + np.array(hard_color) * (1 - hard_alpha)
            hard_rgb = np.clip(hard_rgb, 0, 255).astype(np.uint8)

            # Convert back to PIL Image
            hard_rgb_img = Image.fromarray(hard_rgb)
            hard_rgb_frames.append(hard_rgb_img)

        # Add hard-rendered video to data dict
        data["hard_rgb_video"] = hard_rgb_frames
        data["hard_color"] = hard_color

        return data


class LoadRGBAVideoPair(DataProcessingOperator):
    """
    Load paired RGB and alpha video files for RGBA training.
    Expects paired files like: 1_rgb.mp4 + 1_alpha.mp4

    Based on Wan-Alpha paper requirements for RGBA video generation.
    """

    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor

    def get_num_frames(self, reader):
        """Calculate valid number of frames based on constraints"""
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while (
                num_frames > 1
                and num_frames % self.time_division_factor
                != self.time_division_remainder
            ):
                num_frames -= 1
        return num_frames

    def __call__(self, data: str):
        """
        Load paired RGB and alpha videos.

        Args:
            data: Base path without suffix (e.g., "1" for "1_rgb.mp4" and "1_alpha.mp4")
                  OR path to RGB video (will auto-detect alpha video)

        Returns:
            dict: Dictionary containing 'rgb_video' and 'alpha_video' as lists of PIL Images
        """
        # Determine RGB and alpha paths
        if data.endswith("_rgb.mp4"):
            rgb_path = data
            alpha_path = data.replace("_rgb.mp4", "_alpha.mp4")
        elif data.endswith(".mp4"):
            # Assume it's the base name, try common patterns
            base_path = data.replace(".mp4", "")
            rgb_path = f"{base_path}_rgb.mp4"
            alpha_path = f"{base_path}_alpha.mp4"
        else:
            # Assume data is base path without extension
            rgb_path = f"{data}_rgb.mp4"
            alpha_path = f"{data}_alpha.mp4"

        # Check if files exist
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB video not found: {rgb_path}")
        if not os.path.exists(alpha_path):
            raise FileNotFoundError(f"Alpha video not found: {alpha_path}")

        # Load RGB video
        rgb_reader = imageio.get_reader(rgb_path)
        num_frames = self.get_num_frames(rgb_reader)
        rgb_frames = []
        for frame_id in range(num_frames):
            frame = rgb_reader.get_data(frame_id)
            frame = Image.fromarray(frame).convert("RGB")
            frame = self.frame_processor(frame)
            rgb_frames.append(frame)
        rgb_reader.close()

        # Load alpha video (grayscale/BW)
        alpha_reader = imageio.get_reader(alpha_path)
        alpha_frames = []
        for frame_id in range(num_frames):
            frame = alpha_reader.get_data(frame_id)
            # Convert to single channel grayscale first
            if len(frame.shape) == 3:
                frame = frame[:, :, 0]  # Take first channel if RGB
            frame = Image.fromarray(frame).convert("L")  # Ensure grayscale

            # Duplicate alpha channel three times (as per Wan-Alpha paper)
            # This converts grayscale alpha to 3-channel format matching RGB
            frame = Image.merge("RGB", [frame, frame, frame])

            frame = self.frame_processor(frame)
            alpha_frames.append(frame)
        alpha_reader.close()

        return {"rgb_video": rgb_frames, "alpha_video": alpha_frames}


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = (
            {} if special_operator_map is None else special_operator_map
        )
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920 * 1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
    ):
        return RouteByType(
            operator_map=[
                (
                    str,
                    ToAbsolutePath(base_path)
                    >> LoadImage()
                    >> ImageCropAndResize(
                        height,
                        width,
                        max_pixels,
                        height_division_factor,
                        width_division_factor,
                    ),
                ),
                (
                    list,
                    SequencialProcess(
                        ToAbsolutePath(base_path)
                        >> LoadImage()
                        >> ImageCropAndResize(
                            height,
                            width,
                            max_pixels,
                            height_division_factor,
                            width_division_factor,
                        )
                    ),
                ),
            ]
        )

    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920 * 1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
    ):
        return RouteByType(
            operator_map=[
                (
                    str,
                    ToAbsolutePath(base_path)
                    >> RouteByExtensionName(
                        operator_map=[
                            (
                                ("jpg", "jpeg", "png", "webp"),
                                LoadImage()
                                >> ImageCropAndResize(
                                    height,
                                    width,
                                    max_pixels,
                                    height_division_factor,
                                    width_division_factor,
                                )
                                >> ToList(),
                            ),
                            (
                                ("gif",),
                                LoadGIF(
                                    num_frames,
                                    time_division_factor,
                                    time_division_remainder,
                                    frame_processor=ImageCropAndResize(
                                        height,
                                        width,
                                        max_pixels,
                                        height_division_factor,
                                        width_division_factor,
                                    ),
                                ),
                            ),
                            (
                                ("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
                                LoadVideo(
                                    num_frames,
                                    time_division_factor,
                                    time_division_remainder,
                                    frame_processor=ImageCropAndResize(
                                        height,
                                        width,
                                        max_pixels,
                                        height_division_factor,
                                        width_division_factor,
                                    ),
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )

    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)

    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, "r") as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        result = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        result = self.main_data_operator(data[key])

                    # If operator returns a dict, merge it with data
                    if isinstance(result, dict):
                        data.update(result)
                    else:
                        data[key] = result
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat

    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
