import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_LENGTH = 32

def scale(out, dim=-1, rmax=1, rmin=0):
    out_max = out.max(dim)[0].unsqueeze(dim)
    out_min = out.min(dim)[0].unsqueeze(dim)
    '''
        out_max = out.max()
        out_min = out.min()
    Note that the above max/min is incorrect when batch_size > 1
    '''
    output_std = (out - out_min) / (out_max - out_min)
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled

def is_valid(module, mode = "cov"):
    if mode == "cov":
        return (isinstance(module, nn.Linear)
                or isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Conv1d)
                or isinstance(module, nn.Conv3d)
                or isinstance(module, nn.RNN)
                or isinstance(module, nn.LSTM)
                or isinstance(module, nn.GRU)
                )
    else:
        return (isinstance(module, nn.Sequential)
                or isinstance(module, nn.AdaptiveAvgPool2d)
                or isinstance(module, nn.Linear)
                )

def iterate_module(name, module, name_list, module_list, mode = "cov"):
    if is_valid(module, mode):
        return name_list + [name], module_list + [module]
    else:

        if len(list(module.named_children())):
            for child_name, child_module in module.named_children():
                name_list, module_list = \
                    iterate_module(child_name, child_module, name_list, module_list)
        return name_list, module_list

def get_model_layers(model, mode = "cov"):
    layer_dict = {}
    name_counter = {}
    for name, module in model.named_children():
        name_list, module_list = iterate_module(name, module, [], [], mode)
        assert len(name_list) == len(module_list)
        for i, _ in enumerate(name_list):
            module = module_list[i]
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            layer_dict['%s-%d' % (class_name, name_counter[class_name])] = module
    # DEBUG
    # print('layer name')
    # for k in layer_dict.keys():
    #     print(k, ': ', layer_dict[k])
    return layer_dict


def get_layer_output_sizes(model, data, pad_length=PAD_LENGTH, mode="cov"):

    output_sizes = {}
    hooks = []
    name_counter = {}
    if mode=="cov":
        layer_dict = get_model_layers(model)
    else:
        layer_dict = get_model_layers(model, mode="nac")
    def hook(module, input, output):
        class_name = module.__class__.__name__
        if class_name not in name_counter.keys():
            name_counter[class_name] = 1
        else:
            name_counter[class_name] += 1
        if ('RNN' in class_name) or ('LSTM' in class_name) or ('GRU' in class_name):
            output_sizes['%s-%d' % (class_name, name_counter[class_name])] = [output[0].size(2)]
        else:
            output_sizes['%s-%d' % (class_name, name_counter[class_name])] = list(output.size()[1:])

    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))
    try:
        model(data)
    finally:
        for h in hooks:
            h.remove()

    unrolled_output_sizes = {}
    for k in output_sizes.keys():
        if ('RNN' in k) or ('LSTM' in k) or ('GRU' in k):
            for i in range(pad_length):
                unrolled_output_sizes['%s-%d' % (k, i)] = output_sizes[k]
        else:
            unrolled_output_sizes[k] = output_sizes[k]
    return unrolled_output_sizes

def get_layer_output(model, data, pad_length=PAD_LENGTH, layer_selection = None):

    with torch.no_grad():
        name_counter = {}        
        layer_output_dict = {}
        layer_dict = get_model_layers(model)

        def hook(module, input, output):
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            if ('RNN' in class_name) or ('LSTM' in class_name) or ('GRU' in class_name):
                layer_output_dict['%s-%d' % (class_name, name_counter[class_name])] = output[0]
            else:
                layer_output_dict['%s-%d' % (class_name, name_counter[class_name])] = output

        hooks = []
        for layer, module in layer_dict.items():
            hooks.append(module.register_forward_hook(hook))
        try:
            final_out = model(data)
        finally:
            for h in hooks:
                h.remove()
        if layer_selection is not None:
            layer_output_dict = filter_dict_by_keys(layer_selection, layer_output_dict)
        unrolled_layer_output_dict = {}
        for k in layer_output_dict.keys():
            if ('RNN' in k) or ('LSTM' in k) or ('GRU' in k):
                assert pad_length == len(layer_output_dict[k])
                for i in range(pad_length):
                    unrolled_layer_output_dict['%s-%d' % (k, i)] = layer_output_dict[k][i]
            else:
                unrolled_layer_output_dict[k] = layer_output_dict[k]

        for layer, output in unrolled_layer_output_dict.items():
            if len(output.size()) == 4: # (N, K, H, w)
                output = output.mean((2, 3))
                # _, K, H, W = output.size()
                # if (K, H, W) == (256, 8, 8):
                #     kernel_size = (4, 4)
                #     stride = (4, 4)
                # elif (K, H, W) == (512, 4, 4):
                #     kernel_size = (2, 2)
                #     stride = (2, 2)
                # else:
                #     raise ValueError("Unsupported tensor shape.")

                # # Apply maxpooling
                # if "Conv" in layer:
                #     output = F.max_pool2d(output, kernel_size=kernel_size, stride=stride)
                # output = output.view(output.size(0), -1)
            unrolled_layer_output_dict[layer] = output.detach()
        return unrolled_layer_output_dict

def get_layer_output_nac(model, data, layer_selection=None, use="AL", dataset="cifar10", sigmoids=None):
    """
    Retrieves the output of specified layers in a neural network model.

    Args:
        model (nn.Module): The neural network model.
        data (torch.Tensor): The input data.
        layer_selection (list): List of layer names to select. Default is None.
        use (str): The usage mode. Default is "AL".
        dataset (str): The dataset name. Default is "cifar10".
        sigmoids (dict): Dictionary of sigmoid values for each layer. Default is None.

    Returns:
        dict: Dictionary containing the output of selected layers.
        torch.Tensor: The final output of the model.
    """
    name_counter = {}
    layer_output_dict = {}
    layer_dict = get_model_layers(model, mode="nac")

    def hook(module, input, output):
        class_name = module.__class__.__name__
        if class_name not in name_counter.keys():
            name_counter[class_name] = 1
        else:
            name_counter[class_name] += 1
        layer_output_dict['%s-%d' % (class_name, name_counter[class_name])] = output

    hooks = []
    for layer, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))
    try:
        if use == "AL":
            final_out, _, _ = model(data)
        else:
            final_out = model(data)
    finally:
        for h in hooks:
            h.remove()
    layer_selection = ["AdaptiveAvgPool2d-1", "Sequential-3", "Sequential-2", "Sequential-1", "Linear-1"]
    if layer_selection is not None:
        layer_output_dict = filter_dict_by_keys(layer_selection, layer_output_dict)
    unrolled_layer_output_dict = {}

    for k in layer_output_dict.keys():
        unrolled_layer_output_dict[k] = layer_output_dict[k]
    if dataset == "cifar10":
        sig_alpha = {"AdaptiveAvgPool2d-1": 100, "Sequential-3": 1000, "Sequential-2": 0.001, "Sequential-1": 0.001}
    elif dataset == "cifar100":
        sig_alpha = {"AdaptiveAvgPool2d-1": 50, "Sequential-3": 10, "Sequential-2": 1, "Sequential-1": 0.005}
    elif dataset == "imagenet":
        sig_alpha = {"AdaptiveAvgPool2d-1": 3000, "Sequential-3": 300, "Sequential-2": 0.01, "Sequential-1": 1}
    if sigmoids:
        sig_alpha = sigmoids

    for layer, output in unrolled_layer_output_dict.items():
        if len(output.size()) == 4:  # (N, K, H, w)
            retain_graph = False if "layer1" in layer else True
            output_kl_grad = kl_grad(output, final_out, retain_graph=retain_graph)
            output = output.mean((2, 3))
            output_kl_grad = output_kl_grad.mean((2, 3))
            output = sigmoid(output * output_kl_grad, sig_alpha=sig_alpha[layer])

        unrolled_layer_output_dict[layer] = output.detach()
    return unrolled_layer_output_dict, final_out
     
def filter_dict_by_keys(input_list, input_dict):
    sub_dict = {key: input_dict[key] for key in input_list if key in input_dict}
    return sub_dict

def sigmoid(x, sig_alpha=1.0):
    """
    sig_alpha is the steepness controller (larger denotes steeper)
    """
    return 1 / (1 + torch.exp(-sig_alpha * x))

def get_energy_score(logits):
    """
    Calculates the energy score for the given logits.

    Parameters:
        logits (torch.Tensor): The logits tensor.

    Returns:
        torch.Tensor: The energy score tensor.
    """
    return -torch.logsumexp(logits, dim=1)

def kl_grad(b_state, outputs, temperature=1.0, retain_graph=False, **kwargs):
    """
    This implementation follows https://github.com/deeplearning-wisc/gradnorm_ood
    """
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    num_classes = outputs.shape[-1]
    targets = torch.ones_like(outputs) / num_classes

    loss = (torch.sum(-targets * logsoftmax(outputs), dim=-1))
    layer_grad = torch.autograd.grad(loss.sum(), b_state, create_graph=False,
                                     retain_graph=retain_graph, **kwargs)[0]
    return layer_grad

if __name__ == '__main__':
    pass
