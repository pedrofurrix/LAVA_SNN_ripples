import numpy as np
from enum import Enum
from typing import Callable

def train_printer(
        epoch, iter_counter,
        loss_val, test_loss_val,
        train_acc, test_acc,
        train_f1=None, test_f1=None
    ):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_val:.2f}")
    print(f"Test Set Loss: {test_loss_val:.2f}")
    print(f"Train set accuracy for a single minibatch: {train_acc*100:.2f}%")
    print(f"Test set accuracy for a single minibatch: {test_acc*100:.2f}%")
    if train_f1 is not None and test_f1 is not None:
        print(f"Train set F1 Score for a single minibatch: {train_f1:.2f}")
        print(f"Test set F1 Score for a single minibatch: {test_f1:.2f}")
    print("\n")

def test_printer(
        iter_counter,
        loss_val, acc_val
):
    '''
    Print the test results for a single iteration
    '''
    print(f"Iteration {iter_counter}")
    print(f"Test Set Loss: {loss_val:.2f}")
    print(f"Test set Accuracy: {acc_val*100:.2f}%")
    print("\n")


def undersample_majority(input_data, gt, c1_mask = None) -> tuple[np.ndarray, np.ndarray]:
    '''
    Class Balancing Technique: Under-sample the majority class
    Only implemented for binary classification (2-classes: 0 and 1)

    Parameters:
    - input_data: Input Data.
        Shape: (num_samples, input_shape)
    - gt: Ground Truth.
        Shape: (num_samples, gt_shape).
    - c1_mask: Mask for the C1 class. If None, it will be calculated.
        Shape: (num_samples,).

    Returns:
    - balanced_input_data: Balanced Input Data
    - balanced_gt: Balanced Ground Truth
    '''

    if c1_mask is None:
        # Ripples: gt != -1, Non-ripples: gt == -1
        non_ripple_indices = np.where(gt == -1)[0]
        ripple_indices = np.where(gt != -1)[0]
    else:
        non_ripple_indices = np.where(c1_mask == 0)[0]
        ripple_indices = np.where(c1_mask == 1)[0]

    # Get the minimum commmon number of samples
    num_samples = min(len(non_ripple_indices), len(ripple_indices))   # Number of samples to keep from each class

    # Randomly sort the indices of the two classes
    np.random.shuffle(non_ripple_indices)
    np.random.shuffle(ripple_indices)

    # Get the balanced indices
    balanced_indices = np.concatenate((non_ripple_indices[:num_samples],ripple_indices[:num_samples]))

    # Sort the indices
    balanced_indices = np.sort(balanced_indices)

    # Get the balanced data
    balanced_input_data = input_data[balanced_indices]
    balanced_gt = gt[balanced_indices]

    # Return the balanced data
    return balanced_input_data, balanced_gt

class PerturbationType(Enum):
    '''
    Types of Perturbations to apply to the synthetic samples
    '''
    SHIFT_LEFT = 0,     # Shift the whole sample to the left (circular shift)
    FLIP_BIT = 1,       # Flip a random bit in the sample
    SWITCH_BITS = 2      # Switch 2 random bits in the sample

def oversample_minority(
        input_data, gt,
        c1_mask = None, input_to_gt = None,
        max_sampling_rate: int = 2,
        check_min_class: Callable[[np.array], int] = None,                         
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Class Balancing Technique: Oversample the minority class
    Only implemented for binary classification (2-classes: 0 and 1)
    Oversampling the minority class using Input Perturbations.
    Currently, assumes the input_data is composed of binary vectors (one-hot encoded)

    Parameters:
    ------------
    - input_data: Input Data
        Shape: (num_samples, input_shape)
    - gt: Ground Truth
        Shape: (num_samples, gt_shape)
    - c1_mask: Mask for the C1 class. If None, it will be calculated.
        Shape: (num_samples,)
    - input_to_gt: Function to Calculate the Ground Truth of a Sample
        Parameters:
            - sample: Input Sample
    - max_sampling_rate: Maximum Multiplication Factor of samples for the minority class
        Default: 2 * num_minority_samples. This means the minority class will at most double its samples
    - check_min_class: Function to check if a perturbed sample belongs to the minority class
        Parameters:
            - sample: Perturbed Sample
        Returns:
            - class_id:int Class ID of the sample
    Returns:
    ------------
    - balanced_input_data: Balanced Input Data
    - balanced_gt: Balanced Ground Truth
    '''
    # Get the indices of the two classes
    if c1_mask is None:
        # Get the indices of the two classes
        class_0_indices = np.where(gt == 0)[0]
        class_1_indices = np.where(gt == 1)[0]
    else:
        class_0_indices = np.where(c1_mask == 0)[0]
        class_1_indices = np.where(c1_mask == 1)[0]

    # Get the maximum number of samples of a specific class
    class_to_oversample = 0 if len(class_0_indices) < len(class_1_indices) else 1
    num_minority_samples = min(len(class_0_indices), len(class_1_indices))
    num_majority_samples = max(len(class_0_indices), len(class_1_indices))

    # Define the resulting number of samples of each class
    num_samples_per_class = min(max_sampling_rate * num_minority_samples, num_majority_samples)
    # Define the majority class indices to keep
    majority_indices = class_0_indices if class_to_oversample == 1 else class_1_indices
    # Randomly select the majority indices to keep 
    majority_indices = np.random.choice(majority_indices, num_samples_per_class, replace=False)

    # Get the input data and ground truth of the class to oversample
    minority_indices = class_0_indices if class_to_oversample == 0 else class_1_indices
    minority_input_data = input_data[minority_indices]

    # Calculate the number of samples to oversample
    num_new_samples = num_samples_per_class - len(minority_indices)

    if num_new_samples <= 0:
        # No need to oversample
        return input_data, gt

    # Create the new samples
    synthetic_samples = []

    created_samples = 0
    while created_samples < num_new_samples:
        # Randomly select a sample from the minority class
        rand_idx = np.random.randint(0, len(minority_indices))
        synthetic_sample = np.copy(minority_input_data[rand_idx])

        # Perform some random perturbations to the sample
        num_perturbations = np.random.randint(1, 20)    # Number of perturbations to apply
        for _j in range(num_perturbations):
            curr_perturbation = np.random.choice(list(PerturbationType))
            if curr_perturbation == PerturbationType.SHIFT_LEFT:
                # Shift the sample to the left
                synthetic_sample = np.roll(synthetic_sample, -1)
            elif curr_perturbation == PerturbationType.FLIP_BIT:
                # Flip a random bit in the sample
                rand_bit_idx = np.random.randint(0, len(synthetic_sample))
                synthetic_sample[rand_bit_idx] = 1 - synthetic_sample[rand_bit_idx]
            elif curr_perturbation == PerturbationType.SWITCH_BITS:
                # Switch 2 random bits in the sample

                # Randomly select 2 bits
                rand_bit_idx_1 = np.random.randint(0, len(synthetic_sample))
                rand_bit_idx_2 = None
                while rand_bit_idx_2 is None or rand_bit_idx_2 == rand_bit_idx_1:
                    rand_bit_idx_2 = np.random.randint(0, len(synthetic_sample))

                # Switch the bits
                synthetic_sample[rand_bit_idx_1], synthetic_sample[rand_bit_idx_2] = synthetic_sample[rand_bit_idx_2], synthetic_sample[rand_bit_idx_1]
        
        # Check if the synthetic sample still belongs to the minority class
        # if not, skip the sample
        if check_min_class(synthetic_sample) != class_to_oversample:
            # This sample does not belong to the minority class anymore
            continue

        # Add the synthetic sample
        synthetic_samples.append(synthetic_sample)

        # Increment the number of created samples
        created_samples += 1

    # Merge the minority class and majority class indices
    keep_indices = np.concatenate((majority_indices, minority_indices), axis=0)
    # Sort the indices
    keep_indices = np.sort(keep_indices)

    # Get the input data and ground truth samples to keep without the new synthetic samples
    keep_input_data = input_data[keep_indices]
    keep_gt = gt[keep_indices]

    # Merge the synthetic samples with the existing inputs
    balanced_input_data = np.concatenate((keep_input_data, synthetic_samples), axis=0)

    # Create the ground truth for the new samples
    balanced_gt = np.copy(keep_gt)
    if input_to_gt is None:
        balanced_gt = np.concatenate((keep_gt, np.full((len(synthetic_samples),), class_to_oversample)), axis=0)
    else:
        for new_sample_idx, new_sample in enumerate(synthetic_samples):
            new_gt = input_to_gt(new_sample)

            # Add the new ground truth
            balanced_gt = np.concatenate((balanced_gt, np.array([new_gt])), axis=0)

    # Note: Not shuffling the data -> New samples are added at the end    

    # Return the balanced data
    return balanced_input_data, balanced_gt

