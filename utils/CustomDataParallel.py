from torch.nn.parallel import DataParallel


# Experimental, might be some bugs, but should work for the current set-up in exp-main and data_loader.
# Both this and utils.graph_utils.split_torch_graph should ideally be changed to be more general

def scatter(inputs, target_gpus):
    return [[ob[i] for ob in inputs] for i in range(len(target_gpus))]


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)

    return inputs, kwargs


class DataParallelGraph(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        """
        :param inputs: a list where len(inputs) = num_args, inputs[i] should be a list of length = len(device_ids)
        :param kwargs: a list where len(inputs) = num_kwargs, kwargs[i] should be a list of length = len(device_ids)
        :param device_ids: the gpus to perform distributed training over
        :return: the inputs and kwargs as lists of length = len(device_ids), i.e. len(return[i])=len(device_ids) and len(return[i][j])=num_vars
        """
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)