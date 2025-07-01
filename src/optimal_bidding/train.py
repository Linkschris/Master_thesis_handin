from data import data_loader
from saving import save_results
from model import HindsightModel


def train_model(
    model,
    train_set,
    configfile_name: str,
    saving: bool = False,
    hyperparameter_position: int = None,
    CVaR_position: int = None,
    dataset_size: str = "small",
):
    # run all models in a for loop
    print(f"Received CVaR_position: {CVaR_position}")
    print(f"Received hyperparameter_position: {hyperparameter_position}")
    train_set_copy = train_set.copy()
    model_object = model(
        data=train_set_copy,
        hyperparameter_position=hyperparameter_position,
        CVaR_position=CVaR_position,
        configfile_name=configfile_name,
    )

    print(f"Now the model is {model.__name__}")

    return_dict = model_object.run_model()

    if saving:
        save_results(
            return_dict,
            model.__name__,
            configfile_name,
            dataset_size=dataset_size,
            mode="train",
        )

    return return_dict


if __name__ == "__main__":
    data_size = "small"  # "full" or "small"
    if data_size == "full":
        train_set, val_set, test_set = data_loader("data/processed/data.csv")
    else:
        train_set, val_set, test_set = data_loader("data/processed/data_small.csv")

    configfile_name = "kernel"
    # long_list = [HindsightModel, CVaRModel, PolicyModel]non_linear_PolicyModel
    short_list = [HindsightModel]

    for model in short_list:
        train_model(model, train_set, configfile_name, saving=True, dataset_size=data_size)
