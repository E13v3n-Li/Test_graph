from torch.nn.utils import parameters_to_vector
import logging
from utils import vector_to_model
from watermark_utils import get_watermark_dataset, generate_masks_topk, generate_masks_graph, masks_to_vector
import torch.nn as nn
import torch.optim as optim
import torch
import copy
import builtins


class Server():
    def __init__(self, client_data_sizes, args, main_task_test_loader):
        self.client_data_sizes = client_data_sizes
        self.args = args
        self.main_task_test_loader = main_task_test_loader
        self.best_acc = -1
        self.best_veri = -1

        # ============================================================
        # NEW: state needed for graph-signal-energy-based importance scoring
        # ============================================================
        # Rolling history of each client's per-round parameter update
        # (Delta_i, restricted to named_parameters(), NOT the full
        # state_dict which also contains buffers). Only populated when
        # args.importance_method in ('graph', 'combined').
        self.delta_history = {i: [] for i in range(args.num_clients)}

        # Flat vector of the model that was broadcast to clients at the
        # START of the current round, restricted to named_parameters().
        # This must be set once before the first aggregation call -- see
        # Server.set_initial_broadcast(), which main.py must call right
        # after building initial_model. Without this, round-1 deltas cannot
        # be computed because the server never otherwise sees the
        # pre-training weights (by the time aggregation() is called, all
        # client models have already been locally trained).
        self._last_broadcast_param_vec = None

        # EMA-smoothed client relation adjacency matrix (n x n). Only
        # meaningful when importance_method in ('graph', 'combined').
        self.adjacency = None

    # ------------------------------------------------------------------
    # NEW: must be called once, right after the initial global model is
    # created in main.py, e.g.:
    #     server.set_initial_broadcast(initial_model)
    # ------------------------------------------------------------------
    def set_initial_broadcast(self, initial_model):
        if self.args.watermarking_method == 'tramark' and self.args.importance_method in ('graph', 'combined'):
            self._last_broadcast_param_vec = parameters_to_vector(
                [p.detach() for _, p in initial_model.named_parameters()]
            ).cpu()

    def aggregation(self, client_model_dict, round):
        
        if self.args.watermarking_method == 'fedavg':    
            aggregated_model_dict = self.fedavg(client_model_dict, round)

        if self.args.watermarking_method == 'tramark':
            # If in the warmup stage, we simply perform FedAvg to get a relatively good initial global model for mask generation. 
            if round < int(self.args.training_rounds * self.args.alpha):
                return self.fedavg(client_model_dict, round)
            
            # Right after the warmup stage, we:
            # 1. initialize the masks based on the current global model. 
            # 2. get backdoor dataset for each client.
            if round == int(self.args.training_rounds * self.args.alpha):
                current_global_model = copy.deepcopy(client_model_dict[0])
                self.tramark_initialization(current_global_model)
            
            aggregated_model_dict = self.agg_tramark(client_model_dict, round)
            
        return aggregated_model_dict



    def fedavg(self, client_model_dict, round=None):
        """ classic fed avg: Weighted average based on data size."""

        # ============================================================
        # NEW: record each client's parameter-only update (Delta_i) before
        # aggregating, so that we can later build the client relation graph
        # and compute per-parameter signal energy. This only runs during
        # warmup (fedavg is only called during warmup for watermarking_method
        # == 'tramark'), and only if the importance method actually needs it.
        # ============================================================
        need_delta_history = (
            self.args.watermarking_method == 'tramark'
            and self.args.importance_method in ('graph', 'combined')
            and self._last_broadcast_param_vec is not None
        )
        if need_delta_history:
            for client_id, local_model in client_model_dict.items():
                local_param_vec = parameters_to_vector(
                    [p.detach() for _, p in local_model.named_parameters()]
                ).cpu()
                delta_i = local_param_vec - self._last_broadcast_param_vec
                hist = self.delta_history[client_id]
                hist.append(delta_i)
                if len(hist) > self.args.importance_window:
                    hist.pop(0)

        sm_updates, total_data = 0, 0
        for client_id, local_model in client_model_dict.items():
            local_vector = parameters_to_vector(
                    [local_model.state_dict()[name] for name in local_model.state_dict()]
                ).detach()
            n_agent_data = self.client_data_sizes[client_id]
            sm_updates += local_vector * n_agent_data  
            total_data += n_agent_data

        sm_updates /= total_data
        
        old_model = copy.deepcopy(local_model)
        new_model = vector_to_model(sm_updates, old_model)  

        aggregated_models_dict = {}
        for client_id, local_model in client_model_dict.items():
            aggregated_models_dict[client_id] = copy.deepcopy(new_model)

        # NEW: remember what gets broadcast next round, restricted to
        # named_parameters(), for next round's delta computation.
        if self.args.watermarking_method == 'tramark' and self.args.importance_method in ('graph', 'combined'):
            self._last_broadcast_param_vec = parameters_to_vector(
                [p.detach() for _, p in new_model.named_parameters()]
            ).cpu()

        # Evaluate the aggregated global model on the main task test set
        accuracy = self.test_function(new_model, self.main_task_test_loader)
        logging.info(f"Main task accuracy of the aggregated global model at round {round}: {accuracy:.2f}%")

        if self.best_acc < accuracy:
            self.best_acc = accuracy
        return aggregated_models_dict


    def agg_tramark(self, client_model_dict, round):

        # Masked aggregation: we only aggregate the parameters in the main task region and keep the parameters in the watermarking task region unchanged for each client.
        aggregated_vector = torch.zeros_like(self.main_task_mask_flat, dtype=torch.float32).cuda()
        total_data = 0
        for client_id, model in client_model_dict.items():
            local_vector = parameters_to_vector(
                    [model.state_dict()[name] for name in model.state_dict()]
                ).detach()
            n_agent_data = self.client_data_sizes[client_id]
            aggregated_vector += local_vector * n_agent_data  
            total_data += n_agent_data

        aggregated_vector /= total_data
        aggregated_models_dict = {}

        for client_id, model in client_model_dict.items():
            local_vector = parameters_to_vector(
                        [model.state_dict()[name] for name in model.state_dict()]
                    ).detach()

            ########### Key: Masked aggregation ###########
            mask_aggregated_vector = self.main_task_mask_flat * aggregated_vector + self.watermarking_mask_flat * local_vector

            old_model = copy.deepcopy(model)
            new_model = vector_to_model(mask_aggregated_vector, old_model)
            aggregated_models_dict[client_id] = new_model


        ########### Watermark Injection ###########
        print('Watermarking process starts')
        main_task_acc = []
        for id, local_model in aggregated_models_dict.items():
            pre_inject_acc = self.test_function(local_model, self.main_task_test_loader)
    
            optimizer = optim.SGD(local_model.parameters(), lr=self.watermarking_lr, momentum=self.watermarking_momentum)

            local_model.train()
            for epoch in range(self.watermarking_epochs):
                
                for inputs, targets in self.class_loaders[id]['train']:
                    
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    outputs = local_model(inputs)
                    
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    
                    loss = self.criterion(logits, targets)
                    loss.backward()

                    ########### Zeroing out gradients in the main task region ###########
                    for name, param in dict(local_model.named_parameters()).items():
                        if name in self.watermarking_mask:
                            if param.grad is not None:
                                param.grad.data.mul_(self.watermarking_mask[name].cuda())

                    optimizer.step()

                print(f"Client {id}, watermarking epoch {epoch + 1}/5 finished.")
            post_injection_acc = self.test_function(local_model, self.main_task_test_loader)
            logging.info(f"Client %d, accuracy changes before and after injection: %.2f%% --> %.2f%%" % (id, pre_inject_acc, post_injection_acc))
            main_task_acc.append(post_injection_acc)


        logging.info('-----'*4)

        verification = []
        current_acc = sum(main_task_acc) / len(main_task_acc)
        logging.info('--------Model leaker verification--------')

        for id, local_model in aggregated_models_dict.items():
            local_model.eval()

            current_acc_trigger = []
            for i in range(self.args.num_clients):
                acc = self.test_function(local_model, self.class_loaders[i]['test'])
                current_acc_trigger.append(builtins.round(acc, 2))

            max_index = current_acc_trigger.index(max(current_acc_trigger))

            if max_index == id:
                verification.append(1)
                logging.info(f"Client id: {id}, accuracy on trigger set are {current_acc_trigger}, verification success!")
            else:
                verification.append(0)
                logging.info(f"Client id: {id}, accuracy on trigger set are {current_acc_trigger}, verification fail!")

        current_veri = sum(verification) / len(verification) * 100
        logging.info(f'Round {round} Avg VR: {current_veri:.2f}%')
        logging.info(f'Round {round} Avg ACC: {current_acc:.2f}%')
        if self.best_acc < current_acc:
            self.best_acc = current_acc
        if self.best_veri < current_veri:
            self.best_veri = current_veri
                
        return aggregated_models_dict


    def tramark_initialization(self, current_model):
        # Set up the watermarking hyperparameters
        self.watermarking_epochs = 5
        self.watermarking_lr = 0.0001
        self.watermarking_momentum = 0.0
        
        self.class_loaders = get_watermark_dataset(self.args)
        self.criterion = nn.CrossEntropyLoss()

        # ============================================================
        # NEW: choose region-selection method based on args.importance_method
        # ============================================================
        if self.args.importance_method == 'magnitude':
            # Original TraMark behavior, unchanged.
            self.main_task_mask, self.watermarking_mask = generate_masks_topk(
                partition_ratio=self.args.k, model=current_model)
        else:
            assert len(self.delta_history[0]) >= 1, (
                "No client delta history was collected. Did you forget to call "
                "server.set_initial_broadcast(initial_model) right after creating "
                "the initial model in main.py? Also check that "
                "alpha * training_rounds >= importance_window."
            )
            energy = self._build_graph_and_energy()
            use_magnitude = (self.args.importance_method == 'combined')
            self.main_task_mask, self.watermarking_mask = generate_masks_graph(
                partition_ratio=self.args.k,
                model=current_model,
                energy_normalized=energy,
                combine_lambda=self.args.combine_lambda,
                use_magnitude=use_magnitude)

        self.main_task_mask_flat = masks_to_vector(self.main_task_mask).cuda()
        self.watermarking_mask_flat = masks_to_vector(self.watermarking_mask).cuda()

        # Free the delta history, it is no longer needed once the masks are
        # frozen (masks never change again for the rest of training, exactly
        # like the original TraMark design).
        self.delta_history = {i: [] for i in range(self.args.num_clients)}

    # ============================================================
    # NEW: build the client relation graph from recent delta history and
    # compute the (normalized) Dirichlet energy of each parameter's
    # cross-client signal on that graph.
    # ============================================================
    def _build_graph_and_energy(self):
        n = self.args.num_clients

        # Average each client's most recent `importance_window` deltas to
        # reduce mini-batch noise before computing similarity/energy.
        # Shape: n x d, kept on CPU because d can be tens of millions for
        # VGG16/ViT and storing n copies on GPU may be wasteful.
        S = torch.stack([
            torch.stack(self.delta_history[i]).mean(dim=0) for i in range(n)
        ])  # n x d, float32, CPU

        # Client relation graph: cosine similarity between averaged updates.
        # Only keep non-negative similarity as edge weight -- negative
        # correlation (clients pulling in opposite directions) is not a
        # meaningful "closeness" signal for this purpose.
        S_norm = torch.nn.functional.normalize(S, dim=1, eps=1e-8)
        cos_sim = S_norm @ S_norm.T
        A = cos_sim.clamp(min=0.0)
        A.fill_diagonal_(0.0)  # no self-loops

        if self.adjacency is None:
            self.adjacency = A
        else:
            beta = self.args.graph_ema_beta
            self.adjacency = beta * self.adjacency + (1 - beta) * A

        D = torch.diag(self.adjacency.sum(dim=1))
        L = D - self.adjacency  # graph Laplacian, n x n

        # Dirichlet energy of each parameter's cross-client signal:
        #   energy(j) = s_j^T L s_j / (s_j^T s_j)
        # Computed for all d parameters at once via matrix ops.
        LS = L @ S                        # n x d
        energy = (LS * S).sum(dim=0)      # d
        norm = (S * S).sum(dim=0) + 1e-8  # d
        energy_normalized = energy / norm

        return energy_normalized

    # ============================================================
    # NEW: diagnostic helper. With the current (module B only) design, the
    # watermarking_mask is shared identically across all clients, so overlap
    # is trivially 100%. This is a placeholder / reminder for when module A's
    # graph-coloring (per-client disjoint watermark regions) is added later --
    # do NOT assume this file already implements per-client mask separation.
    # ============================================================
    def log_mask_overlap(self):
        logging.info(
            "NOTE: watermarking_mask is currently shared identically across "
            "all clients (module B only changes WHICH parameters are chosen, "
            "not whether each client gets a different physical region). "
            "Per-client disjoint watermark regions (graph coloring) is a "
            "separate, not-yet-implemented change."
        )

    def test_function(self, model, dataloader):
        """Test the model with the given dataloader."""
        model.eval()
        
        total, correct = 0, 0
        with torch.no_grad():
        
            for inputs, targets in dataloader:
                inputs = inputs.cuda()
                targets = targets.cuda()  

                outputs = model(inputs)
                
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = 100. * correct / total

        return accuracy
