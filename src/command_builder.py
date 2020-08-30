class CommandBuilder:
    def __init__(self, config):
        self.config = config

    def build(self):
        base_cmd = "python3.7 neural-style/neural_style.py  --progress-plot "
        base_cmd += "--network neural-style/imagenet-vgg-verydeep-19.mat "

        base_cmd += f"--content {self.config['content_image_name']} "

        base_cmd += f"--styles {self.config['style_image_name']} "

        output_path = f"{self.config['output_image_name']}"
        base_cmd += f"--output {output_path} "

        # base_cmd += f"--checkpoint-output {self.config['checkpoint_path']} "

        # base_cmd += f"--checkpoint-iteration {self.config['checkpoint_iterations']} "

        if self.config['initial_image']:
            base_cmd += f"--initial {self.config['content_image_name']} "

        base_cmd += f"--iterations {self.config['iterations']} "

        base_cmd += f"--width {self.config['image_width']} "

        base_cmd += f"--learning-rate {self.config['learning_rate']} "

        base_cmd += f"--style-weight {self.config['style_weight']} "

        if self.config['preserve_colors']:
            base_cmd += "--preserve-colors "

        base_cmd += f"--pooling {self.config['pooling']} "

        return base_cmd



