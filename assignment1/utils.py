import os
from typing import TypedDict
import numpy as np
import matplotlib.pyplot as plt
import json


def output_dir(name: str) -> str:
    """Creates output directory if it does not exist."""
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(this_file_dir, "output", name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def save_accuracies_plot(
    output_dir: str, val_accuracies: list[float], test_accuracy: float
):
    """Saves a results plot including validation accuracies and final test accuracy."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save validation accuracies plot
    plt.figure(figsize=(8, 6))

    # Plot validation accuracies
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker="o")
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.grid()

    # Annotate where the best model was found
    # And its test and validation accuracies
    best_epoch = int(np.argmax(val_accuracies) + 1)
    best_val_accuracy = val_accuracies[best_epoch - 1]
    plt.annotate(
        f"Best Model\nEpoch: {best_epoch}\nVal Acc: {best_val_accuracy*100:.2f}%\nTest Acc: {test_accuracy*100:.2f}%",
        xy=(best_epoch, best_val_accuracy),
        xytext=(best_epoch, best_val_accuracy - 0.2),
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="center",
    )

    plt.savefig(os.path.join(output_dir, "accuracies.png"))
    plt.close()


def save_loss_plot(output_dir: str, training_losses: list[float]):
    """Saves the training loss plot."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save training loss plot
    plt.figure(figsize=(8, 6))
    # Plot training loss
    plt.plot(range(1, len(training_losses) + 1), training_losses, marker="o")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()


class LoggingDict(TypedDict):
    training_losses: list[float]
    learning_rate: float
    batch_size: int
    hidden_dims: list[int]
    epochs: int
    seed: int
    use_batch_norm: bool | None


def save_results(
    output_dir: str,
    val_accuracies: list[float],
    test_accuracy: float,
    logging_dict: LoggingDict,
):
    """Saves the losses and accuracies to a json file."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = {
        "val_accuracies": val_accuracies,
        "test_accuracy": test_accuracy,
        "training_losses": logging_dict["training_losses"],
        "learning_rate": logging_dict["learning_rate"],
        "batch_size": logging_dict["batch_size"],
        "hidden_dims": logging_dict["hidden_dims"],
        "epochs": logging_dict["epochs"],
        "seed": logging_dict["seed"],
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(data, f, indent=4)
