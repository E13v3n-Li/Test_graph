import argparse

def get_parse_args():
    parser = argparse.ArgumentParser(description='pass in a parameter')
    
    # System settings
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_path', type=str, default='/home/jiahaox/TraMark/data', help='the path to the dataset')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--client_sampling_ratio', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)

    # Local training settings
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--local_bs', type=int, default=64)
    parser.add_argument('--local_lr', type=float, default=0.01)

    # Non-iid setting
    parser.add_argument('--non_iid', action='store_true', default=False, help='whether to simulate non-iid data distribution, if set to True, please set --gamma')
    parser.add_argument('--gamma', type=float, default=0.5, help='the parameter for non-iid data distribution, the smaller the more non-iid')
    
    # TraMark settings
    parser.add_argument('--watermarking_method', type=str, default='tramark', help='the watermarking method to be used')
    parser.add_argument('--alpha', type=float, default=0.1, help='the fraction of warmup training rounds')
    parser.add_argument('--k', type=float, default=0.01, help='the partition ratio for the main task region and the watermarking task region')

    # ============================================================
    # NEW: Graph-signal-energy-based parameter importance settings
    # ============================================================
    parser.add_argument('--importance_method', type=str, default='magnitude',
                         choices=['magnitude', 'graph', 'combined'],
                         help="how to score parameter importance for region partitioning. "
                              "'magnitude': original TraMark (|theta|). "
                              "'graph': pure client-graph signal energy (cross-client disagreement). "
                              "'combined': rank-weighted combination of magnitude and graph energy.")
    parser.add_argument('--importance_window', type=int, default=5,
                         help='number of most recent rounds of client updates (delta_i) used to '
                              'estimate the client relation graph and signal energy. Ignored if '
                              'importance_method == magnitude. NOTE: alpha * training_rounds must '
                              'be >= importance_window, or there will not be enough history collected '
                              'by the time region partitioning happens.')
    parser.add_argument('--graph_ema_beta', type=float, default=0.9,
                         help='EMA smoothing coefficient for the client relation adjacency matrix '
                              '(only used to smooth the graph across the accumulation window, not '
                              'across the whole training run).')
    parser.add_argument('--combine_lambda', type=float, default=0.7,
                         help="weight on the magnitude ranking when importance_method == 'combined'. "
                              "score = lambda * magnitude_rank + (1-lambda) * consensus_rank. "
                              "Start high (0.7-0.8) so magnitude still dominates and graph energy only "
                              "nudges the decision, rather than fully replacing it.")

    args = parser.parse_args()
    
    return args
