config_template = {}
config_template["content_image_name"] = None
config_template["style_image_name"] = None
config_template["experiment_name"] = None
config_template["output_image_name"] = None
config_template["experiment_root"] = None
config_template["iterations"] = 12
config_template["image_width"] = 500
config_template["learning_rate"] = 1.0
config_template["style_weight"] = 1000_000
config_template["content_weight"] = 1

search_config_template = {}
search_config_template["iterations"] = "3-30"
search_config_template["learning_rate"] = "0.01 - 30.0"
search_config_template["style_weight"] = "100000 - 2000000"
search_config_template["content_weight"] = "0.1-100.0"
