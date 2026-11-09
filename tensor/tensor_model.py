from model.losses import detection_loss
from model.network import create_person_detection
from prepare_dataset import prepare_samples


if __name__ == "__main__":
    import runpy

    runpy.run_module("train", run_name="__main__")
