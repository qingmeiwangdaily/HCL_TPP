import json
import os
import torch

SAVED_MODELS_PATH = "saved_models"


def save_model(model: torch.nn.Module, chosen_data_file, extra_tag, hidden_size, now_timestamp, model_name=None):
    if model_name is None:
        model_name = model.__class__.__name__
    filename_base = "{}-{}_hidden{}-{}".format(
        model_name, extra_tag,
        hidden_size, now_timestamp)
    filename_model_save = filename_base + ".pth"
    model_filepath = os.path.join(SAVED_MODELS_PATH, filename_model_save)
    print("Saving models to: {}".format(model_filepath))
    torch.save(model.state_dict(), model_filepath)

    file_correspondance = {
        "model_path": model_filepath,
        "data_path": chosen_data_file
    }
    print(file_correspondance)

    with open(os.path.join(SAVED_MODELS_PATH, "train_data_correspondance.jsonl"), "a") as f:
        json.dump(file_correspondance, f)
        f.write('\n')

