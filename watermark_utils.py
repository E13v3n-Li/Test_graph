from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torch


def create_class_specific_datasets(dataset, num_train_per_class=None, num_clients=None):

    class_specific_datasets = {}
    class_indices = {label: [] for label in range(num_clients)}

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    for label in range(num_clients):
        if num_train_per_class is None:
            selected_indices = class_indices[label]
        else:
            selected_indices = class_indices[label][:num_train_per_class]

        class_specific_datasets[label] = Subset(dataset, selected_indices)

    return class_specific_datasets


def create_class_specific_loaders(train_dataset, test_dataset, num_train_per_class, batch_size=32, num_workers=2, num_clients=10):
    
    train_class_datasets = create_class_specific_datasets(train_dataset, num_train_per_class, num_clients=num_clients)
    test_class_datasets = create_class_specific_datasets(test_dataset, num_train_per_class=None, num_clients=num_clients) 

    loaders = {
        label: {
            'train': DataLoader(train_class_datasets[label], batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test': DataLoader(test_class_datasets[label], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }
        for label in range(num_clients)
    }

    return loaders


def get_watermark_dataset(args):

    # Get the appropriate transformations for the mnist dataset based on the main dataset
    if args.dataset == 'tiny':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])
    elif args.dataset == 'fmnist':
        transform_mnist = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
    elif args.dataset == 'cifar10':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif args.dataset == 'cifar100':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])


    train_mnist = torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transform_mnist)
    test_mnist = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform_mnist)
    
    class_loaders = create_class_specific_loaders(train_mnist, 
                                                  test_mnist, 
                                                  num_train_per_class=100, 
                                                  batch_size=32, 
                                                  num_workers=args.num_workers, 
                                                  num_clients=args.num_clients)
    

    return class_loaders


def generate_masks_topk(partition_ratio, model):
    """
    Original TraMark region selection: purely based on parameter magnitude
    (|theta|). Kept unchanged so that --importance_method magnitude
    reproduces the exact original behavior.
    """

    partition_ratio = 1 - partition_ratio
    keys = [name for name in model.state_dict()]
    keys_parameters = [name for name, param in model.named_parameters()]
    
    param_dict = {name: param for name, param in model.named_parameters()}
    main_task_mask = {}
    watermarking_mask = {}

    for name in keys:
        if name in keys_parameters:  
            param = param_dict[name]
            numel = param.numel()
            mask = torch.zeros(numel, dtype=torch.bool)

            flat_param = param.view(-1).abs()
            topk = int(numel * partition_ratio)
            if topk > 0:
                _, selected_indices = torch.topk(flat_param, topk, largest=True)
                mask[selected_indices] = True 

        else:
            param = model.state_dict()[name]
            numel = param.numel()
            mask = torch.ones(numel, dtype=torch.bool)  

        main_task_mask[name] = mask.view_as(param)
        watermarking_mask[name] = ~mask.view_as(param)  

    return main_task_mask, watermarking_mask


# ============================================================
# NEW: graph-signal-energy-based region selection
# ============================================================
def generate_masks_graph(partition_ratio, model, energy_normalized, combine_lambda=0.7, use_magnitude=True):
    """
    Select the main-task region / watermarking region using a client-graph
    signal energy score instead of (or combined with) raw parameter magnitude.

    Rationale (see discussion): for parameter j, energy_normalized[j] is the
    normalized Dirichlet energy of the "client update signal"
        s_j = (Delta_1[j], ..., Delta_n[j])
    on the client relation graph. A LOW energy means all clients (especially
    graph neighbors) agree on how this parameter should move -> likely a
    shared/main-task feature -> should stay in the main task region. A HIGH
    energy means clients disagree strongly -> likely client-specific ->
    a reasonable (not required, but low-cost) place to put the watermark.

    Args:
        partition_ratio: k, the fraction of parameters assigned to the
            watermarking region (same meaning as in generate_masks_topk).
        model: current global model (post-warmup snapshot).
        energy_normalized: 1D tensor whose length equals the total number of
            elements across model.named_parameters() (NOT state_dict()!),
            in the exact same flattening order as
            torch.nn.utils.parameters_to_vector(model.named_parameters()).
            This is produced by Server._build_graph_and_energy().
        combine_lambda: weight on the magnitude ranking when
            use_magnitude=True. score = lambda * mag_rank + (1-lambda) * consensus_rank.
        use_magnitude: if True, blend with magnitude (recommended default,
            corresponds to --importance_method combined). If False, use pure
            graph energy (corresponds to --importance_method graph).

    Returns:
        main_task_mask, watermarking_mask: same dict-of-boolean-tensors
        format as generate_masks_topk, so the rest of the pipeline
        (masks_to_vector, agg_tramark, etc.) does not need to change.
    """
    k = partition_ratio
    keys = [name for name in model.state_dict()]
    keys_parameters = [name for name, param in model.named_parameters()]
    param_dict = {name: param for name, param in model.named_parameters()}

    # Slice the flat energy vector back into per-parameter-tensor chunks,
    # following the exact same order as named_parameters().
    pointer = 0
    energy_dict = {}
    for name in keys_parameters:
        numel = param_dict[name].numel()
        energy_dict[name] = energy_normalized[pointer:pointer + numel]
        pointer += numel

    assert pointer == energy_normalized.numel(), (
        f"energy_normalized length ({energy_normalized.numel()}) does not match "
        f"the total number of elements in model.named_parameters() ({pointer}). "
        f"Check that it was built with parameters_to_vector(model.named_parameters()), "
        f"not state_dict()."
    )

    main_task_mask, watermarking_mask = {}, {}

    for name in keys:
        if name in keys_parameters:
            param = param_dict[name]
            numel = param.numel()

            flat_mag = param.view(-1).abs()
            flat_energy = energy_dict[name].to(flat_mag.device)

            # Rank-normalize both signals to [0, 1] so that they are on a
            # comparable scale regardless of the raw units of magnitude vs.
            # energy (they otherwise have completely different scales, and a
            # raw weighted sum would just be dominated by whichever has the
            # larger numeric range).
            if numel > 1:
                mag_rank = flat_mag.argsort().argsort().float() / (numel - 1)
                energy_rank = flat_energy.argsort().argsort().float() / (numel - 1)
            else:
                mag_rank = torch.zeros_like(flat_mag)
                energy_rank = torch.zeros_like(flat_energy)

            # High energy = high cross-client disagreement -> LOW consensus.
            # We want HIGH consensus_rank to mean "keep in main task region",
            # matching the convention that high mag_rank means "important,
            # keep in main task region" in the original method.
            consensus_rank = 1.0 - energy_rank

            if use_magnitude:
                score = combine_lambda * mag_rank + (1.0 - combine_lambda) * consensus_rank
            else:
                score = consensus_rank

            mask = torch.zeros(numel, dtype=torch.bool)
            topk = int(numel * (1 - k))
            if topk > 0:
                _, selected_indices = torch.topk(score, topk, largest=True)
                mask[selected_indices] = True

        else:
            # Buffers (e.g. BatchNorm running_mean/running_var,
            # num_batches_tracked) are not part of named_parameters() and
            # therefore have no energy score. Keep the original convention:
            # always assign buffers to the main task region.
            param = model.state_dict()[name]
            numel = param.numel()
            mask = torch.ones(numel, dtype=torch.bool)

        main_task_mask[name] = mask.view_as(param)
        watermarking_mask[name] = ~mask.view_as(param)

    return main_task_mask, watermarking_mask


def masks_to_vector(masks):

    vectors = []
    for name, mask in masks.items():
        vectors.append(mask.flatten())  

    return torch.cat(vectors)
