import torch
import torch.nn as nn
from snntorch import spikegen

# Use labels by default unless target_is_time = True


class SpikeTimePenalty(nn.Module):
    """
    Used by ce_temporal_loss and mse_temporal_loss to convert spike
    outputs into spike times including a penalty for extreme cases.
    Extreme cases:
    1. When the output neuron does not spike when it should have spiked.
    2. When the output neuron spikes when it should not have spiked.
    In such cases, the spike time instead of being set to the final time step,
    is set to: final time step * penalty_factor.
    """

    def __init__(
        self,
        target_is_time=False,
        on_target=0,
        off_target=-1,
        tolerance=0,
        multi_spike=False,
        # Penalty Factor for Extreme Cases (Default: 1 for no penalty)
        penalty_factor=1.0,
    ):
        super().__init__()

        self.target_is_time = target_is_time
        self.tolerance = tolerance
        self.tolerance_fn = self.Tolerance.apply
        self.multi_spike = multi_spike
        self.penalty_factor = penalty_factor

        if not self.target_is_time:
            self.on_target = on_target
            self.off_target = off_target  # override this with final step

        # function used to extract the first F spike times. If
        # multi_spike=False, F=1.
        if self.multi_spike:
            self.first_spike_fn = self.MultiSpike.apply
        else:
            self.first_spike_fn = self.FirstSpike.apply

    # spiking output from final layer is a recording: T x B x N
    # targets can either be labels or spike times
    def forward(self, spk_out, targets):
        '''
        Parameters:
        - spk_out: Spike Output (Binary Spikes for each time step)
            Shape: (num_steps, batch_size, num_outputs)
        - targets: Target Labels (First Spike Times)
            Shape: (batch_size, num_outputs)
        '''
        self.device, num_steps, num_outputs = self._prediction_check(spk_out)

        # convert labels to spike times
        if not self.target_is_time:
            targets = self.labels_to_spike_times(targets, num_outputs)

        '''
        Convert negative spikes times -> No spike (-1) -> (num_steps - 1) * penalty_factor
        This converts the GT events where the output neuron should not spike.
        This is done by adding a "shadow spike" at the last time step of the window.
        '''
        targets[targets < 0] = ((spk_out.size(0) - 1) * self.penalty_factor) 

        # now operating in the spike-time domain rather than with labels
        # Consider merging multi-spike and single-spike?
        # single-spike is faster, so keep them separate for now.
        if self.multi_spike:
            self.spike_count = targets.size(0)
            spk_time_final = self.first_spike_fn(
                spk_out, self.spike_count,
            )  # spk_time_final here means the first spike time
        else:
            # spk_time_final = self.first_spike_fn(spk_out, self.device)
            spk_time_final = self.first_spike_fn(
                spk_out, self.penalty_factor,
            )

        '''
        Check if any output neuron has GT != -1 near the end of the window ]num_steps-tolerance-1, num_steps-1]
        and observed spikes = 0. If so, divide the observed spike time by the penalty factor since the neuron should not have spiked.
        '''
        # Check if any output neuron has GT = 1 near the end of the window
        targets_near_end_mask = (
            ((num_steps - 1 - self.tolerance) < targets) & (targets < num_steps) 
        )
        # define mask for output neurons that did not spike
        spk_time_no_spike_mask = spk_time_final >= num_steps

        # If GT is near the end and the neuron did not spike, set the spk_time to num_steps (Revert the penalty factor)
        spk_time_final[targets_near_end_mask & spk_time_no_spike_mask] = num_steps - 1

        # next need to check how tolerance copes with multi-spikes
        if self.tolerance:
            spk_time_final = self.tolerance_fn(
                spk_time_final, targets, self.tolerance
            )

        return spk_time_final, targets

    def _prediction_check(self, spk_out):
        # device = "cpu"
        # if spk_out.is_cuda:
        #     device = "cuda"
        device = spk_out.device

        num_steps = spk_out.size(0)
        num_outputs = spk_out.size(-1)

        return device, num_steps, num_outputs

    @staticmethod
    class FirstSpike(torch.autograd.Function):
        """
        Convert spk_rec of 1/0s [TxBxN] --> first spike time [BxN].
        Linearize df/dS=-1 if spike, 0 if no spike for backpropagation loss.
        """
        @staticmethod
        def forward(ctx, spk_rec, penalty_factor):
            """
            Convert spk_rec of 1/0s [TxBxN] --> spk_time [TxBxN].
            0's indicate no spike --> +1 is first time step.
            Transpose accounts for broadcasting along final dimension
            (i.e., multiply along T).

            Parameters:
            - spk_rec: Spike Recordings (Binary Spikes for each time step)
                Shape: (num_steps, batch_size, num_outputs)
            - device: Device (CPU or CUDA)
            - penalty_factor: Penalty Factor for Extreme Cases
                (Default: 1 for no penalty)
            """
            # Define the device
            device = spk_rec.device

            # spk_time is a tensor of the same shape as spk_rec (num_steps, batch_size, num_outputs)
            # where each element is the time step at which the spike occurred + 1
            # For example, if spk_rec[3, 0, 0] = 1, then spk_time[3, 0, 0] = 4
            spk_time = (
                spk_rec.transpose(0, -1)
                * (torch.arange(0, spk_rec.size(0)).detach().to(device) + 1)
            ).transpose(0, -1)

            """extract first spike time. Will be used to pass into loss
            function."""
            num_steps = spk_time.size(0)
            # First Spike Time shape: (batch_size, num_outputs)
            first_spike_time = torch.zeros_like(spk_time[0])
            for step in range(num_steps):
                # Add the spike time to first_spike_time only if it is not already set
                # This ensures that only the first spike time is recorded for each neuron
                first_spike_time += (
                    spk_time[step] * ~first_spike_time.bool()
                )  # mask out subsequent spikes

            # Offset the first spike time by -1 for neurons that have a spike > 0 (i.e., they spiked)
            first_spike_time = first_spike_time - (first_spike_time.bool() * 1)  # fix offset for spiking neurons

            '''
            Add Context for backward pass before modifying
            spike_time with penalty factor since the time step
            would fall out of bounds (non-differentiable)
            NOTE: This means that for the output neurons that did not spike,
            the gradient will be set to -1 in the last time step, but the
            penalized_spike_times will return a value of (num_steps - 1) * penalty_factor
            in the forward pass
            ''' 
            # Create a MASK for the neurons that did not spike (i.e., first_spike_time == 0)
            # This will be used to update the first spike time to the last time step for the backward
            # pass and later update it to last_time_step * penalty_factor
            SILENT_NEURONS_MASK = first_spike_time == 0
            # Set the first spike time to the last time step for those neurons
            first_spike_time[SILENT_NEURONS_MASK] = num_steps - 1

            # Save tensors for backward pass
            # first_spike_time: (batch_size, num_outputs)
            # spk_rec: (num_steps, batch_size, num_outputs)
            ctx.save_for_backward(first_spike_time, spk_rec)

            '''
            At this point, first_spike_time contains the first spike time for each neuron
            If there is no spike, it will be at the last timestep. So now we need to multiply
            those neurons by the penalty factor to add the penalty to the loss
            '''
            penalized_spike_times = first_spike_time.clone()
            # Set the penalized spike times for the neurons that did not spike
            penalized_spike_times[SILENT_NEURONS_MASK] *= penalty_factor
            # print(f"First Spike Time: {first_spike_time} | Penalized Spike Times: {penalized_spike_times}")

            return penalized_spike_times

        @staticmethod
        def backward(ctx, grad_output):
            (first_spike_time, spk_rec) = ctx.saved_tensors
            spk_time_grad = torch.zeros_like(spk_rec)  # T x B x N

            """spike extraction step/indexing @ each step is
            non-differentiable.
            Apply sign estimator by substituting gradient for -1 ONLY at
            first spike time."""
            for batch_idx in range(first_spike_time.size(0)):
                for output_idx in range(first_spike_time.size(1)):
                    # Set the gradient to -1 at the first spike time of each output neuron of each batch.
                    # This means that increasing the membrane potential will cause the neuron to spike earlier.
                    # Gradient is only defined for that one time step.
                    # TODO: Could this be done differently?
                    spk_time_grad[first_spike_time[batch_idx,
                                                   output_idx].long(), batch_idx, output_idx] = 1.0

            grad = -grad_output * spk_time_grad
            return grad, None

    @staticmethod
    class MultiSpike(torch.autograd.Function):
        """Convert spk_rec of 1/0s [TxBxN] --> first F spike times [FxBxN].
        Linearize df/dS=-1 if spike, 0 if no spike."""

        @staticmethod
        def forward(ctx, spk_rec, spk_count):
            # Define the device
            device = spk_rec.device

            spk_rec_tmp = spk_rec.clone()
            spk_time_rec = []

            for step in range(spk_count):
                """Convert spk_rec of 1/0s [TxBxN] --> spk_time [TxBxN].
                0's indicate no spike --> +1 is first time step.
                Transpose accounts for broadcasting along final dimension
                (i.e., multiply along T)."""
                spk_time = (
                    spk_rec_tmp.transpose(0, -1)
                    * (
                        torch.arange(0, spk_rec_tmp.size(0))
                        .detach()
                        .to(device)
                        + 1
                    )
                ).transpose(0, -1)

                """extact n-th spike time (n=step) up to F."""
                nth_spike_time = torch.zeros_like(spk_time[0])
                for step in range(spk_time.size(0)):
                    nth_spike_time += (
                        spk_time[step] * ~nth_spike_time.bool()
                    )  # mask out subsequent spikes

                """override element 0 (no spike) with shadow spike @ final
                time step, then offset by -1
                s.t. first_spike is at t=0."""
                nth_spike_time += ~nth_spike_time.bool() * (
                    spk_time.size(0)
                )  # populate non-spiking with total size
                nth_spike_time -= 1  # fix offset
                spk_time_rec.append(nth_spike_time)

                """before looping, eliminate n-th spike. this avoids double
                counting spikes."""
                spk_rec_tmp[nth_spike_time.long()] = 0

            """Pass this into loss function."""
            spk_time_rec = torch.stack(spk_time_rec)

            ctx.save_for_backward(spk_time_rec, spk_rec)

            return spk_time_rec

        @staticmethod
        def backward(ctx, grad_output):
            (spk_time_final, spk_rec) = ctx.saved_tensors
            spk_time_grad = torch.zeros_like(spk_rec)  # T x B x N

            """spike extraction step/indexing @ each step is
            non-differentiable.
            Apply sign estimator by substituting gradient for -1 ONLY at
            F-th spike time."""
            for i in range(spk_time_final.size(0)):
                for j in range(spk_time_final.size(1)):
                    for k in range(spk_time_final.size(2)):
                        spk_time_grad[
                            spk_time_final[i, j, k].long(), j, k
                        ] = -grad_output[i, j, k]
            grad = spk_time_grad
            return grad, None, None

    @staticmethod
    class Tolerance(torch.autograd.Function):
        """If spike time is 'close enough' to target spike within tolerance,
        set the time to target for loss calc only."""

        # TO-DO: remove ctx?
        @staticmethod
        def forward(ctx, spk_time, target, tolerance):
            spk_time_clone = (
                spk_time.clone()
            )  # spk_time_clone: BxN (FxBxN for multi-spike); target: TxBxN

            '''
            Set the spike time to the target time if it is within the tolerance
            This is done by checking if the absolute difference between the
            spike time and target is less than the tolerance
            if spk_time in ]target - tolerance, target + tolerance[ (exclusive interval)
            '''
            # Define the mask for neurons that are within the tolerance
            tolerance_mask = torch.abs(spk_time - target) < tolerance

            # Define the mask for neurons that 

            # Set the spike time to the target time for those neurons
            spk_time_clone[tolerance_mask] = (
                torch.ones_like(spk_time) * target
            )[tolerance_mask]

            spk_time_clone[torch.abs(spk_time - target) < tolerance] = (
                torch.ones_like(spk_time) * target
            )[torch.abs(spk_time - target) < tolerance]
            return spk_time_clone

        @staticmethod
        def backward(ctx, grad_output):
            grad = grad_output
            return grad, None, None

    def labels_to_spike_times(self, targets, num_outputs):
        """Convert index labels [B] into spike times."""

        if not self.multi_spike:
            targets = self.label_to_single_spike(targets, num_outputs)

        # pass in labels --> output multiple spikes
        # assumes on_target & off_target are iterable
        else:
            targets = self.label_to_multi_spike(targets, num_outputs)

        return targets

    def label_to_single_spike(self, targets, num_outputs):
        """Convert labels from neuron index (dim: B) to first spike time
        (dim: B x N)."""

        # guess: i designed this code with on_target >> off_target in mind
        targets = spikegen.targets_convert(
            targets,
            num_classes=num_outputs,
            on_target=self.on_target,
            off_target=self.off_target,
        )

        return targets

    def label_to_multi_spike(self, targets, num_outputs):
        """Convert labels from neuron index (dim: B) to multiple spike times
        (dim: F x B x N).
        F is the number of spikes per neuron. Assumes target is iterable
        along F."""

        num_spikes_on = len(self.on_target)
        num_spikes_off = len(self.off_target)

        if num_spikes_on != num_spikes_off:
            raise IndexError(
                f"`on_target` (length: {num_spikes_on}) must have the same "
                f"length as `off_target` (length: {num_spikes_off}."
            )

        # iterate through each spike
        targets_rec = []
        for step in range(num_spikes_on):
            target_step = spikegen.targets_convert(
                targets,
                num_classes=num_outputs,
                on_target=self.on_target[step],
                off_target=self.off_target[step],
            )
            targets_rec.append(target_step)
        targets_rec = torch.stack(targets_rec)

        return targets_rec


class mse_temporal_loss_penalty():
    """Mean Square Error Temporal Loss with a Penalty for Extreme Cases.

    Extreme cases:
    1. When the output neuron does not spike when it should have spiked.
    2. When the output neuron spikes when it should not have spiked.
    In such cases, the spike time instead of being set to the final time step, is set to: final time step * penalty_factor.

    The first spike time of each output neuron [batch_size x num_outputs] is
    measured against the desired spike time with the Mean Square Error Loss
    Function.
    Note that the derivative of each spike time with respect to the spike
    df/dU is non-differentiable for most neuron classes, and is set to a sign
    estimator of -1.
    I.e., increasing membrane potential causes a proportionately earlier
    firing time.

    The Mean Square Error Temporal Loss can account for multiple spikes by
    setting ``multi_spike=True``.
    If the actual spike time is close enough to the target spike time within
    a given tolerance, e.g., ``tolerance = 5`` time steps, then it does not
    contribute to the loss.import torch.nn as nn

    Index labels are passed as the target by default.
    To enable passing in the spike time(s) for output neuron(s), set
    ``target_is_time=True``.

    Note: After spike times with specified targets, no penalty is applied
    for subsequent spiking.
    To eliminate later spikes, an additional target should be applied.

    Example::

        import torch
        import snntorch.functional as SF

        # default takes in idx labels as targets
        # correct classes aimed to fire by default at t=0, incorrect at t=-1
        (final time step)
        loss_fn = mse_temporal_loss()
        loss = loss_fn(spk_out, targets)

        # as above, but correct class fire @ t=5, incorrect at t=100 with a
        tolerance of 2 steps
        loss_fn = mse_temporal_loss(on_target=5, off_target=100, tolerance=2)
        loss = loss_fn(spk_out, targets)

        # as above with multiple spike time targets
        on_target = torch.tensor(5, 10)
        off_target = torch.tensor(100, 105)
        loss_fn = mse_temporal_loss(on_target=on_target,
        off_target=off_target, tolerance=2)
        loss = loss_fn(spk_out, targets)

        # specify first spike time for 5 neurons individually, zero tolerance
        target = torch.tensor(5, 10, 15, 20, 25)
        loss_fn = mse_temporal_loss(target_is_time=True)
        loss = loss_fn(spk_out, target)


    :param target_is_time: Specify if target is specified as spike times
        (True) or as neuron indexes (False). Defaults to ``False``
    :type target_is_time: bool, optional

    :param on_target: Spike time for correct classes
        (only if target_is_time=False). Defaults to ``0``
    :type on_target: int
        (or interable over multiple int if ``multi_spike=True``), optional

    :param off_target: Spike time for incorrect classes
        (only if target_is_time=False).
        Defaults to ``-1``, i.e., final time step
    :type off_target: int (or interable over multiple int if
        ``multi_spike=True``), optional

    :param tolerance: If the distance between the spike time and target is
        less than the specified tolerance, then it does not contribute to the
        loss. Defaults to ``0``.
    :type tolerance: int, optional

    :param multi_spike: Specify if multiple spikes in target. Defaults to
        ``False``
    :type multi_spike: bool, optional

    :return: Loss
    :rtype: torch.Tensor (single element)

    """

    def __init__(
        self,
        target_is_time=False,
        on_target=0,
        off_target=-1,
        tolerance=0,
        multi_spike=False,
        reduction='mean',
        weight=None,
        penalty_factor=None,    # Penalty Factor for Extreme Cases
        normalize=True,
    ):
        super().__init__()

        self.reduction = reduction
        self.weight = weight
        self.loss_fn = nn.MSELoss(reduction=(
            'none' if self.weight is not None else self.reduction))
        self.spk_time_fn = SpikeTimePenalty(
            target_is_time, on_target, off_target, tolerance, multi_spike, penalty_factor
        )
        self.__name__ = "mse_temporal_loss"
        self.normalize = normalize

    def __call__(self, spk_rec, targets):
        '''
        Operator called when the class is called as a function - mse_temporal_penalty()

        Parameters:
        - spk_rec: Spike Recordings (Binary Spikes for each time step)
            Shape: (num_steps, batch_size, num_outputs)
        - targets: Target Labels (First Spike Times)
            Shape: (batch_size, num_outputs)
        '''
        spk_time, target_time = self.spk_time_fn(
            spk_rec, targets
        )  # return encoded targets
        # print(f"Spike Time: {spk_time}, Target Time: {target_time}")

        # Calculate the loss using the defined loss function
        # The loss is scaled by the number of time steps (spk_rec.size(0))

        norm_factor = 1.0
        if self.normalize:
            num_steps = spk_rec.size(0)
            norm_factor = num_steps
        
        loss = self.loss_fn(
            spk_time / norm_factor,
            target_time / norm_factor
        )  # spk_time_final: num_spikes x B x Nc. # Same with targets.

        if self.weight is not None:
            loss = loss * self.weight[targets]
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss

def first_spike_acc(output, target, tolerance=0, verbose=False):
    '''
    Accuracy Metric to calculate the accuracy of the first spike prediction
    ---------
    Parameters:
    output : torch.Tensor shape=(num_steps, batch_size, num_classes)
        Spikes of the output neurons.
    target : torch.Tensor shape=(batch_size, num_classes)
        Ground truth of the network - First spike time of each output neuron
    tolerance : int
        Tolerance for the accuracy calculation. If the output neuron spikes within this tolerance, it is considered correct.
    verbose : bool
        Whether to print the output or not
    ---------

    Returns:
    float | np.nan
        The accuracy of the first spike prediction. If no valid predictions are found, return np.nan
    '''
    # Ensure the output and target are on the same device
    output = output.to(target.device)

    num_steps = output.shape[0]

    # --- Calculate the accuracy ---
    # Find the first spike time of each output neuron
    # Shape: (batch_size, num_classes)
    output_first_spike_time = torch.zeros(target.shape, device=target.device)    # Initialize with -1 (No Spike)
    for class_idx in range(target.shape[1]):
        output_first_spike_time[:, class_idx] = torch.argmax(output[:, :, class_idx], dim=0)
    
    # Check if the output neuron spikes at all -> If not, set the first spike time to -1
    output_first_spike_time[torch.sum(output, dim=0) == 0] = -1
    
    if verbose:
        print(f"Output First Spike Time: {output_first_spike_time} | Target: {target}")

    # NOTE: If the GT Spike time is almost at the end of the window, the accuracy is not calculated correctly.
    '''
    Check if any output neuron has GT = 1 near the end of the window ]num_steps-tolerance-1, num_steps-1]
    and observed spikes = 0. If so, the prediction should not be considered into the accuracy calculation,
    '''
    # Check if any output neuron has GT = 1 near the end of the window
    targets_near_end_mask = (
        ((num_steps - 1 - tolerance) < target) & (target < num_steps) 
    )
    # define mask for output neurons that did not spike (equal to -1)
    spk_time_no_spike_mask = (output_first_spike_time == -1)
    # Define mask combining the two masks
    invalid_mask = targets_near_end_mask & spk_time_no_spike_mask
    if verbose:
        print(f"Targets Near End Mask: {targets_near_end_mask}")
        print(f"Spk Time No Spike Mask: {spk_time_no_spike_mask}")
        print(f"Invalid Mask: {invalid_mask}")
    '''
    If GT is near the end and the neuron did not spike, the prediction should not be considered into the
    accuracy calculation, since the window after the GT is < tolerance window Set the GT to -1 as well in those cases
    '''
    
    # Update the output first spike time and target to exclude the invalid predictions
    valid_mask = ~invalid_mask
    output_first_spike_time = output_first_spike_time[~invalid_mask]
    target = target[~invalid_mask]

    # --- Calculate the spike time differences ---
    # Before that, transform the -1 values to the end of the window
    output_first_spike_time[output_first_spike_time == -1] = num_steps - 1
    target[target == -1] = num_steps - 1

    # accuracy = SF.accuracy_temporal(output, target)
    spikeDiffs = torch.abs(output_first_spike_time - target)
    if verbose:
        print(f"Spike Diffs: {spikeDiffs}")

    # Check if the spike time differences are within the tolerance window
    # Shape: (batch_size, num_classes)
    spikeDiffsWithinTolerance = spikeDiffs < tolerance
    
    # Calculate the accuracy
    # Shape: ()
    accuracy = torch.mean(spikeDiffsWithinTolerance.float())

    if verbose:
        print(f"Accuracy Value: {accuracy*100}%\n====================\n")
    
    return accuracy

# TODO: Fix the accuracy calculation