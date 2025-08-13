import re

BAL_EQUAL_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/simple_cot/BAL-Equal_simple.jsonl",
    "data_path": "",
}

CW_ONLY_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/simple_cot/CW-Only_simple.jsonl",
    "data_path": "",
}

NO_WARP_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/simple_cot/No-Warp_simple.jsonl",
    "data_path": "",
}

C3F2W1_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/simple_cot/C3F2W1_simple.jsonl",
    "data_path": "",
}

# Simple Classification datasets
BAL_EQUAL_SIMPLE_CLASS_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/datasets/simple_classification/BAL-Equal_simple_class.jsonl",
    "data_path": "",
}

CW_ONLY_SIMPLE_CLASS_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/datasets/simple_classification/CW-Only_simple_class.jsonl",
    "data_path": "",
}

NO_WARP_SIMPLE_CLASS_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/datasets/simple_classification/No-Warp_simple_class.jsonl",
    "data_path": "",
}

C3F2W1_SIMPLE_CLASS_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/datasets/simple_classification/C3F2W1_simple_class.jsonl",
    "data_path": "",
}

EXPERIMENT_VALIDATION = {
    "annotation_path": "/home/intern/banner_vis/data/validation_test/validation_abs.jsonl",
    "data_path": "",
}

EXPERIMENT_TEST = {
    "annotation_path": "/home/intern/banner_vis/data/validation_test/test_abs.jsonl",
    "data_path": "",
}

data_dict = {
    "bal_equal": BAL_EQUAL_DATASET,
    "cw_only": CW_ONLY_DATASET,
    "no_warp": NO_WARP_DATASET,
    "c3f2w1": C3F2W1_DATASET,
    "bal_equal_simple_class": BAL_EQUAL_SIMPLE_CLASS_DATASET,
    "cw_only_simple_class": CW_ONLY_SIMPLE_CLASS_DATASET,
    "no_warp_simple_class": NO_WARP_SIMPLE_CLASS_DATASET,
    "c3f2w1_simple_class": C3F2W1_SIMPLE_CLASS_DATASET,
    "experiment_validation": EXPERIMENT_VALIDATION,
    "experiment_test": EXPERIMENT_TEST,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["bal_equal"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
