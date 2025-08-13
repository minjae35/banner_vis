import re

BAL_EQUAL_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/BAL-Equal_fixed_abs.jsonl",
    "data_path": "",
}

CW_ONLY_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/CW-Only_fixed_abs.jsonl",
    "data_path": "",
}

NO_WARP_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/No-Warp_fixed_abs.jsonl",
    "data_path": "",
}

C3F2W1_DATASET = {
    "annotation_path": "/home/intern/banner_vis/data/experiments/C3F2W1_fixed_abs.jsonl",
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
