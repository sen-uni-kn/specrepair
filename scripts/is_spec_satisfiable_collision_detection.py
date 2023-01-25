import pandas
import torch
from tqdm import tqdm


if __name__ == "__main__":
    training_data = pandas.read_csv("../resources/collision_detection/CollisionDetection_train_data.csv")
    train_inputs = torch.tensor(training_data.drop('class', axis=1).values, dtype=torch.float)
    train_targets = torch.tensor(training_data['class'].values, dtype=torch.long)

    # collision_detection_repair_4 specification
    # radius = 0.05
    # sample_indices = [4, 5, 6, 8, 10, 11, 13, 16, 18, 19, 20, 22, 23, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 42, 44]

    # whole dataset
    # radius = 0.01  # fine
    radius = 0.05  # inconsistent
    # radius = 0.04  # inconsistent
    # radius = 0.03  # fine
    # radius = 0.035  # fine
    # radius = 0.039  # inconsistent
    sample_indices = list(range(len(train_inputs)))

    print(f"Number of samples: {len(sample_indices)}, eps: {radius}")
    sample_inputs = train_inputs[sample_indices, :]
    sample_targets = train_targets[sample_indices]

    total_contradictions = 0
    for i in tqdm(sample_indices):
        x1 = train_inputs[i, :]
        y1 = train_targets[i]
        is_close = (torch.abs(x1 - sample_inputs) <= radius).all(dim=-1)
        is_contradicting = sample_targets[is_close] != y1
        num_contradictions = sum(is_contradicting)
        total_contradictions += num_contradictions
        if num_contradictions > 0:
            print(f"Data Point {i} is in contradiction with:")
            is_close_counter = 0
            for j in range(len(sample_indices)):
                if is_close[j]:
                    if is_contradicting[is_close_counter]:
                        print(f"  - data point {j}")
                    is_close_counter += 1

    if total_contradictions > 0:
        print("Specification is inconsistent.")
        print(f"{total_contradictions} inconsistencies.")
    else:
        print("Specification is consistent.")
