import collections
from copy import deepcopy
import logging
import datetime
import argparse
import json
import os
from contextlib import contextmanager
import subprocess
from config_parser import parse
from fever_io import read_jsonl, save_jsonl


root_dir = "/scratch/fz758/ucl"
@contextmanager
def environ(env):
    original_environ_dict = os.environ.copy()
    os.environ.update(env)
    yield
    os.environ.clear()
    os.environ.update(original_environ_dict)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def save_config(config, path):
    with open(path, "w") as f:
        json.dump(config, f)


def convert(config):
    # train data
    if not os.path.exists(config["train_converted_file"]):
        options = list()
        options.append(config["train_input_file"])
        options.append(config["train_converted_file"])
        script = ["converter.py"] + options
        __run_python(script, gpu=False, env={"PYTHONPATH": "."})
    else:
        logger.info("%s already exists. skipping conversion for train", config["train_converted_file"])

    # dev data
    if not os.path.exists(config["dev_converted_file"]):
        options = list()
        options.append(config["dev_input_file"])
        options.append(config["dev_converted_file"])
        script = ["converter.py"] + options
        __run_python(script, gpu=False, env={"PYTHONPATH": "."})
    else:
        logger.info("%s already exists. skipping conversion for dev", config["dev_converted_file"])

    # test data
    if ("test_converted_file" in config) and (not os.path.exists(config["test_converted_file"])):
        options = list()
        options.append(config["test_input_file"])
        options.append(config["test_converted_file"])
        script = ["converter.py"] + options
        __run_python(script, gpu=False, env={"PYTHONPATH": "."})
    else:
        if "test_converted_file" in config:
            logger.info("%s already exists. skipping conversion for dev", config["test_converted_file"])


def train_rte(config):
    os.chdir("../jack")
    options = list()
    options.append("with")
    options.append("config={}".format(config["jack_config_file"]))
    options.append("save_dir={}".format(config["save_dir"]))
    options.append("train={}".format(config["train_file"]))
    options.append("dev={}".format(config["dev_file"]))
    if "load_dir" in config and config["load_dir"] != "":
        options.append("load_dir={}".format(config["load_dir"]))

    script = ["bin/jack-train.py"] + options
    __run_python(script, gpu=True, env={"PYTHONPATH": "."})
    os.chdir("../snli")


def inference_rte(config):
    os.chdir(os.path.join(root_dir, "jack"))
    options = list()
    options.append(config["train_input_file"])  # input file
    options.append(config["train_predicted_labels_and_scores_file"])
    options.extend(["--saved_reader", config["saved_reader"]])
    if config["batch_size"]:
        options.extend(["--batch_size", str(config["batch_size"])])

    # train data
    if not os.path.exists(config["train_predicted_labels_and_scores_file"]):
        options_train = options
        if "preprocessed_train_input_file" in config:
            options_train.extend(["--preprocessed_in_file", 
                config["preprocessed_train_input_file"]])
        script = ["../snli/jack_reader.py"] + options_train
        __run_python(script, gpu=True, env={"PYTHONPATH": "."})
    else:
        logger.info("skipping inference rte for train. %s exists", config["train_predicted_labels_and_scores_file"])

    # dev data
    if not os.path.exists(config["dev_predicted_labels_and_scores_file"]):
        options[0] = config["dev_input_file"]
        options[1] = config["dev_predicted_labels_and_scores_file"]
        options_dev = options
        if "preprocessed_dev_input_file" in config:
            options_dev.extend(["--preprocessed_in_file",
                config["preprocessed_dev_input_file"]])
        script = ["../snli/jack_reader.py"] + options_dev
        __run_python(script, gpu=True, env={"PYTHONPATH": "."})
    else:
        logger.info("skipping inference rte for dev. %s exists", config["dev_predicted_labels_and_scores_file"])

    # test data
    if not os.path.exists(config["test_predicted_labels_and_scores_file"]):
        options[0] = config["test_input_file"]
        options[1] = config["test_predicted_labels_and_scores_file"]
        options_test = options
        if "preprocessed_test_input_file" in config:
            options_test.extend(["--preprocessed_in_file",
                config["preprocessed_test_input_file"]])
        script = ["../snli/jack_reader.py"] + options_test
        __run_python(script, gpu=True, env={"PYTHONPATH": "."})
    else:
        logger.info("skipping inference rte for test. %s exists", config["test_predicted_labels_and_scores_file"])
    os.chdir("../snli")


def score(config):
    if not os.path.exists(config["train_score_file"]):
        options = list()
        options.extend(["--prediction_file", config["train_prediction_file"]])
        options.extend(["--score_file", config["train_score_file"]])

        script = ["score.py"] + options
        __run_python(script, gpu=False, env={"PYTHONPATH": "."})
    else:
        logger.info("skipping scoring for train. %s exists", config["train_score_file"])
    
    if not os.path.exists(config["dev_score_file"]):
        options = list()
        options.extend(["--prediction_file", config["dev_prediction_file"]])
        options.extend(["--score_file", config["dev_score_file"]])

        script = ["score.py"] + options
        __run_python(script, gpu=False, env={"PYTHONPATH": "."})
    else:
        logger.info("skipping scoring for dev. %s exists", config["dev_score_file"])
    
    if not os.path.exists(config["test_score_file"]):
        options = list()
        options.extend(["--prediction_file", config["test_prediction_file"]])
        options.extend(["--score_file", config["test_score_file"]])

        script = ["score.py"] + options
        __run_python(script, gpu=False, env={"PYTHONPATH": "."})
    else:
        logger.info("skipping scoring for test. %s exists", config["test_score_file"])

def __run_python(script, gpu=False, env=dict()):
    ### You might need to add env vars or/and python executables.
    # Examples:
    # LD_LIBRARY_PATH = "/share/apps/cuda-9.0/lib64:/share/apps/python-3.6.3-shared/lib:/share/apps/libc6_2.23/lib/x86_64-linux-gnu:/share/apps/libc6_2.23/lib64:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib"
    # python_gpu_prep = [
    #     "/share/apps/libc6_2.23/lib/x86_64-linux-gnu/ld-2.23.so",
    #     "/home/tyoneda/anaconda3/bin/python3"
    # ]
    # prep = ["/home/tyoneda/anaconda3/bin/python3"]

    python_gpu_prep = [
        "python3"
    ]
    prep = ["python3"]
    if gpu:
        # env.update({
        #     "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
        #     # "CUDA_VISIBLE_DEVICES": "0"
        # })
        prep = python_gpu_prep

    with environ(env):
        script = prep + script
        logger.info("running: %s", script)
        ret = subprocess.run(script)
        if ret.returncode != 0:
            logger.info("returned: %s", ret)
            raise RuntimeError("shell returned non zero code.")


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model", default="{0:model_%Y%m%d%H%M%S}".format(now))
    parser.add_argument(
        "--overwrite", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true")
    parser.add_argument(
        "--preprocessing", action="store_true")
    args = parser.parse_args()
    print(args)
    if os.path.exists(os.path.join("results", args.model, "org_config.json")) and not args.overwrite:
        logger.warning("overwriting the existing model due to --overwrite flag.")
        raise RuntimeError("you cannot overwrite the config. use different model name.")

    exec_flag = False

    with open(args.config) as f:
        config = json.load(f)

    # load and save original config
    if "__variables" not in config:
        config["__variables"] = {}
    # config["__variables"]["___model_name___"] = args.model
    model_dir = "results/{}".format(config["__variables"]["___model_name___"])

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    logger.info("model dir: %s", model_dir)
    save_config(config, path=os.path.join(model_dir, "org_config.json"))

    # load child config if specified
    if "parent_config" in config and config["parent_config"]:
        path = config["parent_config"]
        logger.info("loading parent config from {}".format(path))
        with open(path) as f:
            parent_config = json.load(f)
        save_config(parent_config, path=os.path.join(model_dir, "parent_config.json"))
        config = update(deepcopy(parent_config), config)

    config = parse(config)
    save_config(config, path=os.path.join(model_dir, "config.json"))

    # convert format if file does not exist
    conf_convert = config["convert"]
    logger.info("%s exists?: %s", conf_convert["train_converted_file"], os.path.exists(conf_convert["train_converted_file"]))
    logger.info("%s exists?: %s", conf_convert["dev_converted_file"], os.path.exists(conf_convert["dev_converted_file"]))
    logger.info("%s exists?: %s", conf_convert["test_converted_file"], os.path.exists(conf_convert["test_converted_file"]))
    if not( os.path.exists(conf_convert["train_converted_file"]) 
        and os.path.exists(conf_convert["dev_converted_file"])
        and os.path.exists(conf_convert["test_converted_file"])):
        convert(conf_convert)
        exec_flag = True
    else:
        logger.info("skipping conversion...")
    
    if args.preprocessing:
        print("Exited due to preprocessing flag")
        exit()

    # train rte model if file does not exist
    conf_train_rte = config["train_rte"]
    logger.info("%s exists?: %s", conf_train_rte["save_dir"], os.path.exists(conf_train_rte["save_dir"]))
    if exec_flag or (not os.path.isdir(conf_train_rte["save_dir"])):
        train_rte(conf_train_rte)
        exec_flag = True
    else:
        logger.info("skipping train rte...")

    # rte inference if file does not exist
    conf_inference = config["inference_rte"]
    logger.info("%s exists?: %s", conf_inference["train_predicted_labels_and_scores_file"], os.path.exists(conf_inference["train_predicted_labels_and_scores_file"]))
    logger.info("%s exists?: %s", conf_inference["dev_predicted_labels_and_scores_file"], os.path.exists(conf_inference["dev_predicted_labels_and_scores_file"]))
    logger.info("%s exists?: %s", conf_inference["test_predicted_labels_and_scores_file"], os.path.exists(conf_inference["test_predicted_labels_and_scores_file"]))
    if exec_flag or (
        not os.path.exists(conf_inference["train_predicted_labels_and_scores_file"]) or 
        not os.path.exists(conf_inference["dev_predicted_labels_and_scores_file"]) or 
        not os.path.exists(conf_inference["test_predicted_labels_and_scores_file"])):
        inference_rte(config["inference_rte"])
        exec_flag = True
    else:
        logger.info("skipping inference rte...")

    # scoring
    conf_score = config["score"]
    logger.info("%s exists?: %s", conf_score["train_score_file"], os.path.exists(conf_score["train_score_file"]))
    logger.info("%s exists?: %s", conf_score["dev_score_file"], os.path.exists(conf_score["dev_score_file"]))
    logger.info("%s exists?: %s", conf_score["test_score_file"], os.path.exists(conf_score["test_score_file"]))
    if exec_flag or (
        not os.path.exists(conf_score["train_score_file"]) or 
        not os.path.exists(conf_score["dev_score_file"]) or 
        not os.path.exists(conf_score["test_score_file"])):
        score(conf_score)
    else:
        logger.info("skipping score...")