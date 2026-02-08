import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import sklearn.metrics

# Import metric computation functions
from minotaur.training.metrics import compute_concept_metric_single, compute_task_metric_single

# Import losses and attention mechanisms from separate modules
from .losses import CoxLoss
from .attention import MultiKQVAtten, MultiConceptAttention, ConceptAttention


# Linear layer followed by normalization, ReLU activation, and dropout
class LinearBatchNorm(torch.nn.Module):
    """
    Linear layer followed by normalization (batch or instance), ReLU activation, and dropout.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        bt_dim: int = None,
        norm_type: str = 'batch',
        activation = torch.nn.LeakyReLU()
    ) -> None:
        """
        Initializes the LinearBatchNorm block.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            dropout (float): Dropout rate.
            bt_dim (int, optional): Dimension for normalization; defaults to out_features.
            norm_type (str): Type of normalization ('batch' or 'instance').
        """
        super(LinearBatchNorm, self).__init__()
        
        self.norm_type = norm_type
        self.bt_dim = out_features if bt_dim is None else bt_dim
        self.lbn_block = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            activation,
            torch.nn.Dropout(p=dropout),
            self.get_norm()
        )

    def get_norm(self) -> torch.nn.Module:
        """
        Returns the appropriate normalization layer based on `norm_type`.

        Returns:
            nn.Module: BatchNorm1d or InstanceNorm1d layer.
        """
        if self.norm_type == 'batch':
            return torch.nn.BatchNorm1d(self.bt_dim)
        elif self.norm_type == 'instance':
            return torch.nn.InstanceNorm1d(self.bt_dim)
        else:
            raise ValueError(
                f"Unsupported norm type '{self.norm_type}'."
                )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LinearBatchNorm block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, features).

        Returns:
            torch.Tensor: Output tensor after linear, activation, dropout, and normalization.
        """
        return self.lbn_block(x)


# Pre-bottleneck MLP
class PreMLP(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        activation,
        mlp=True 
    ):
        super(PreMLP, self).__init__()
        self.in_dim = in_dim
        self.pre_mlp = torch.nn.Sequential(
            LinearBatchNorm(
                in_dim,
                int(in_dim/2),
                dropout,
            ),
            torch.nn.Linear(int(in_dim/2), out_dim),
            activation
        )
    
    def forward(self, x):
        return self.pre_mlp(x)
    

# Concept context generator
class ConceptContextGenerator(torch.nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            dropout,
            activation,
            mlp=True    
    ):
        super(ConceptContextGenerator, self).__init__()

        self.activation = activation
        if mlp:
            self.ccpt_ctxt_gen = torch.nn.Sequential(
                LinearBatchNorm(
                    in_dim,
                    int(in_dim/2),
                    dropout,
                ),
                torch.nn.Linear(int(in_dim/2), out_dim),
                activation
            )
        else:
            self.ccpt_ctxt_gen = torch.nn.Sequential(*[
                torch.nn.Linear(
                    in_dim,
                    int(in_dim)/2,
                ),
                activation
            ]
            )

    def forward(self, x):
        return self.ccpt_ctxt_gen(x)
    

# Concept probability generator
class ConceptProbGenerator(torch.nn.Module):
    def __init__(self, in_features, out_features=1):
        super(ConceptProbGenerator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.prob_gen = torch.nn.Linear(
            self.in_features,
            self.out_features
        )

    def forward(self, x):
        return self.prob_gen(x)


# Main Concept Embedding Model - merged from CEM and essential CBM methods
class ConceptEmbeddingModel(pl.LightningModule):
    activations = {
        'Sigmoid': torch.nn.Sigmoid(),
        'ReLU': torch.nn.ReLU(),
        'LeakyReLU': torch.nn.LeakyReLU()
    }

    def __init__(
        self,
        n_concepts,
        n_tasks,
        h_dim,
        emb_size=16,
        concept_states=None,
        embedding_activation="LeakyReLU",
        shared_prob_gen=False,
        concept_loss_weight=1,
        task_loss_weight=1,
        task_class_weights=None,
        concept_class_weights=17*[1.0],
        n_att_heads=4,
        attn_dim=256,
        attn_dropout=0.4,
        dropout=0.4,
        pre_bn_mlp=None,
        c2y_model=None,
        c2y_layers=None,
        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        top_k_accuracy=None,
        attention_hook=None,
        no_concepts=False,
        task_type='binary'  # 'binary', 'multiclass', 'continuous', or 'cox'
    ):
        """Concept Embedding Model (CEM) for multiple instance learning with concept embeddings."""
        super().__init__()
        
        # Store parameters
        self.n_concepts = n_concepts
        self.concept_states = concept_states
        self.h_dim = h_dim
        self.emb_size = emb_size
        self.attn_dim = attn_dim
        self.dropout = attn_dropout
        self.n_att_heads = n_att_heads
        self.shared_prob_gen = shared_prob_gen
        self.pre_bn_mlp = pre_bn_mlp
        self.dropout = dropout
        self.task_type = task_type
        self.task_loss_weight = task_loss_weight
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.top_k_accuracy = top_k_accuracy
        self.no_concepts = no_concepts
        
        # Initialize architecture based on no_concepts flag
        if self.no_concepts:
            # No concepts mode: single attention block, single pre-MLP, direct to c2y_model
            self.attention_hook = attention_hook
            if attention_hook is not None:
                self.attention_blocks = torch.nn.ModuleList(
                    [MultiConceptAttention(
                        self.h_dim,
                        self.attn_dim, 
                        self.n_att_heads, 
                        self.dropout
                        )])
                for block in self.attention_blocks:
                    block.register_forward_hook(self.attention_hook.hook_fn)
                ctxt_in_dim = self.n_att_heads * self.h_dim
            else:
                ctxt_in_dim = self.h_dim
                
            self.b_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(ctxt_in_dim)])
        else:
            # Original concept-based mode
            self.attention_hook = attention_hook
            if attention_hook is not None:
                self.attention_blocks = torch.nn.ModuleList(
                    [MultiConceptAttention(
                        self.h_dim,
                        self.attn_dim, 
                        self.n_att_heads, 
                        self.dropout
                        ) for _ in range(n_concepts)])

                for block in self.attention_blocks:
                    block.register_forward_hook(self.attention_hook.hook_fn)
            
                ctxt_in_dim = self.n_att_heads * self.h_dim
            else:
                ctxt_in_dim = self.h_dim
             
            self.b_norms = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(ctxt_in_dim) for _ in range(n_concepts)]
            )
        
        # Pre-bottleneck MLP
        if self.pre_bn_mlp:
            out_dim = int(ctxt_in_dim/4)
            self.pre_mlp = PreMLP(
                 ctxt_in_dim,
                int(ctxt_in_dim/4),
                self.dropout,
                ConceptEmbeddingModel.activations[embedding_activation]
                )   

            self.pre_mlp_reg = PreMLP(
                 ctxt_in_dim,
                int(ctxt_in_dim/4),
                self.dropout,
                ConceptEmbeddingModel.activations[embedding_activation]
                )   
            ctxt_in_dim_v = int(out_dim)
        else:
            ctxt_in_dim_v = ctxt_in_dim

        # Initialize concept-related components only if not in no_concepts mode
        if not self.no_concepts:
            # Concept context generators
            self.concept_context_generators = torch.nn.ModuleList()
            for i in range(n_concepts):
                if self.concept_states[i] > 1:
                    num_states = self.concept_states[i]
                    ctxt_out_dim = num_states * self.emb_size
                elif self.concept_states[i] == 1:
                    num_states = self.concept_states[i]
                    ctxt_out_dim = 2 * self.emb_size
                elif self.concept_states[i] == 0:  # Continuous concept
                    num_states = 1
                    ctxt_out_dim = self.emb_size * 1
                else:
                    num_states = 1
                    ctxt_out_dim = self.emb_size
                self.concept_context_generators.append(
                    ConceptContextGenerator(
                        ctxt_in_dim_v,
                        ctxt_out_dim,
                        self.dropout,
                        ConceptEmbeddingModel.activations[embedding_activation]
                    )
                )
            
            # Concept prob generators
            self.concept_prob_generators = torch.nn.ModuleList()
            for i in range(n_concepts):
                if self.concept_states[i] > 1:
                    num_states = self.concept_states[i]
                    ctxt_out_dim = num_states * self.emb_size
                elif self.concept_states[i] == 1:
                    num_states = self.concept_states[i]
                    ctxt_out_dim = 2 * self.emb_size
                elif self.concept_states[i] == 0:  # Continuous concept
                    num_states = 1
                    ctxt_out_dim = self.emb_size * 1
                else:
                    num_states = 1
                    ctxt_out_dim = self.emb_size
                out_dim = num_states if self.concept_states[i] > 1 else 1
                self.concept_prob_generators.append(ConceptProbGenerator(
                        ctxt_out_dim,
                        out_dim))
                
                if self.shared_prob_gen:
                    break
            
            # Concept attention mechanism - learns importance weights for each concept
            self.concept_attention = ConceptAttention(
                emb_size=self.emb_size,
                attn_dim=64,  # Can be made configurable
                dropout=self.dropout
            )
        else:
            # In no_concepts mode, initialize empty concept-related components
            self.concept_context_generators = torch.nn.ModuleList()
            self.concept_prob_generators = torch.nn.ModuleList()
            self.concept_attention = None
        
        # Post-bottleneck layer
        n_node = 1 if n_tasks == 'continuous' else n_tasks
        
        if c2y_model is None:
            if self.no_concepts:
                # In no_concepts mode, input dimension is the pre-MLP output or ctxt_in_dim
                input_dim = int(ctxt_in_dim/4) if self.pre_bn_mlp else ctxt_in_dim
                units = [input_dim] + (c2y_layers or []) + [n_node]
            else:
                # Original concept-based mode
                units = [
                    n_concepts * emb_size
                ] + (c2y_layers or []) + [n_node]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model
        self.sig = torch.nn.Sigmoid()

        # Concept loss functions - support mixed continuous, binary, and categorical concepts
        self.loss_concept_fns = []
        if not self.no_concepts:
            for i, states in enumerate(self.concept_states):
                if states == 0:
                    # Continuous concept (e.g., RNA_Bio)
                    self.loss_concept_fns.append(torch.nn.MSELoss())
                elif states == 1:
                    # Binary concept
                    self.loss_concept_fns.append(torch.nn.BCEWithLogitsLoss())
                else:
                    # Categorical concept
                    self.loss_concept_fns.append(torch.nn.CrossEntropyLoss())
        
        # Task loss
        if task_type == 'cox':
            self.loss_task = CoxLoss()
        elif n_tasks == 'continuous':
            self.loss_task = torch.nn.MSELoss()
        elif n_tasks > 1:
            self.loss_task = torch.nn.CrossEntropyLoss(
                weight=task_class_weights
                )
        else:
            self.loss_task = torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
                )

    # Attention helper method
    def _get_atten(self, x, atten_block):
        """
        Computes attention score and generates an attention-weighted embedding.

        Args:
            x (torch.Tensor): Input embedding tensor Shape: (batch_size, num_tiles, embedd_dim).
            atten_block (torch.nn.Module): Attention block to compute attention scores.

        Returns:
            h_attn (torch.Tensor): Attention-weighted embedding. Shape: (batch_size, n_att_heads * embedd_dim).
            a_w_prob (torch.Tensor): Average attention probabilities for all tiles Shape: (batch_size, num_tiles).
        """
           
        a_w = atten_block(x) # (batch_size, num_tiles, n_att_heads)
        a_w_avg = a_w.mean(dim=-1)
        sm = torch.nn.Softmax()
        a_w_prob = sm(a_w_avg)
                
        a_w  = torch.transpose(a_w, -1, -2) # (batch_size, n_att_heads, num_tiles)
        h_attn = torch.matmul(a_w, x) # (batch_size, n_att_heads, h_dim)
        h_attn = h_attn.flatten(1, -1)  # (batch_size, n_att_heads * h_dim)
        
        return h_attn, a_w
    

    # Concept state embeddings helper method
    def _get_concept_state_embeddings(self, h, context_gen):
        return context_gen(h)  # (batch_size, num_states * emb_size)

    # Concept state scores helper method
    def _get_concept_state_probs(self, context, cpt_idx):
        prob_gen = self.concept_prob_generators[cpt_idx]
        logits = prob_gen(context)

        if self.concept_states[cpt_idx] == 0:
            # Continuous concept - return raw logits for regression
            probs = logits
        elif self.concept_states[cpt_idx] == 1:
            # Binary concept - apply sigmoid
            probs = torch.nn.Sigmoid()(logits)
        else:
            # Categorical concept - apply softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs, logits

    def _get_concept_embeddings(self, state_probs, contexts):
        # Flexible approach: handle all concept types uniformly
        cpt_embedds = []
        start_idx = 0
        prob_idx = 0
        
        for num_states in self.concept_states:
            if num_states == 0:
                # Continuous concept: use single embedding directly
                cpt_embedd = contexts[:, start_idx, :]
                start_idx += 1
            elif num_states == 1:
                # Binary concept: weighted combination of 2 embeddings
                end_idx = start_idx + 2
                bin_vec = contexts[:, start_idx:end_idx, :]

                pos, neg = torch.chunk(bin_vec, 2, dim=1)
                bin_p = state_probs[:, prob_idx]
                cpt_embedd  = pos * bin_p.view(-1, 1, 1) + neg * (1 - bin_p).view(-1,1,1)
                cpt_embedd = cpt_embedd.squeeze(1) 
                start_idx = end_idx
                prob_idx+=1
            
            else:
                # Categorical concept: weighted combination
                end_idx = start_idx + num_states
                cat_contexts = contexts[:, start_idx:end_idx, :]
                cat_probs = state_probs[:, start_idx:end_idx]
                cpt_embedd = (cat_contexts * cat_probs.unsqueeze(-1)).sum(dim=1)
                start_idx = end_idx
            
            cpt_embedds.append(cpt_embedd)
        
        # Stack concept embeddings to create (batch, n_concepts, emb_size) tensor
        concept_embeddings = torch.stack(cpt_embedds, dim=1)  # (batch, n_concepts, emb_size)
        
        # Apply concept attention if available
        if self.concept_attention is not None:
            concept_attention_weights, weighted_concept_embeddings = self.concept_attention(concept_embeddings)
            # Flatten weighted embeddings for final prediction
            final_embeddings = weighted_concept_embeddings.view(weighted_concept_embeddings.shape[0], -1)
            return final_embeddings, concept_attention_weights
        else:
            # Fallback to concatenation if no concept attention
            return torch.cat(cpt_embedds, dim=1), None 

    # Main forward pass - MIL and categorical context generation logic
    def _forward(
        self,
        x,
        c=None,
        y=None,
        train=False,
    ):
        h = x
        
        if self.no_concepts:
            # No concepts mode: single attention block, single pre-MLP, direct to c2y_model
            if self.attention_hook is not None:
                atten_block = self.attention_blocks[0]  # Single attention block
                h_attn, a_w_prob = self._get_atten(h, atten_block)
                h_slide = self.b_norms[0](h_attn)  # Single batch norm
            else:
                h_slide = h.mean(dim=1)
                a_w_prob = None
            
            if self.pre_bn_mlp:
                h_slide = self.pre_mlp(h_slide)
            
            # Direct prediction without concept processing
            y = self.c2y_model(h_slide)
            
            # Return None for concept-related outputs
            return tuple([None, None, y, None, [a_w_prob], None, None])
        else:
            # Original concept-based mode
            contexts = []
            state_probs = []
            state_logits = []
            a_w_prob_lst = []
        
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.attention_hook is not None:
                    atten_block = self.attention_blocks[i]
                    h_attn, a_w_prob = self._get_atten(h, atten_block)
                    h_slide = self.b_norms[i](h_attn)
                else:
                    h_slide = h.mean(dim=1)
                
                if self.pre_bn_mlp:
                    if self.concept_states[i]==0:
                        h_slide = self.pre_mlp_reg(h_slide)
                    else:
                        h_slide = self.pre_mlp(h_slide)
                         
                state_context = self._get_concept_state_embeddings(h_slide, context_gen)
                probs, logits = self._get_concept_state_probs(state_context, i)

                if self.concept_states[i] == 0:
                    # Continuous or binary concept - reshape to (batch_size, 1, emb_size)
                    state_context = torch.unsqueeze(state_context, dim=1)
                elif self.concept_states[i] == 1:
                    state_context = state_context.view(
                        state_context.shape[0], 2, self.emb_size
                    )
                else:
                    # Categorical concept
                    num_states = self.concept_states[i]
                    state_context = state_context.view(
                        state_context.shape[0], num_states, self.emb_size
                        )  # (batch_size, num_states, emb_size)

                contexts.append(state_context) # (batch_size, num_states, emb_size)
                state_probs.append(probs) # (batch_size, num_states)
                state_logits.append(logits) # (batch_size, num_states)
                a_w_prob_lst.append(a_w_prob)

            contexts = torch.cat(contexts, dim=1) # (batch_size, sum(concept_states), emb_size)
            state_probs = torch.cat(state_probs, dim=1) # (batch_size, sum(concept_states))
            state_logits = torch.cat(state_logits, dim=1) # (batch_size, sum(concept_states))
            cpt_embedds, concept_attention_weights = self._get_concept_embeddings(state_probs, contexts)
            y = self.c2y_model(cpt_embedds)
            
            return tuple([state_logits, cpt_embedds, y, self.concept_states, a_w_prob_lst, contexts, concept_attention_weights])

    def forward(
        self,
        x,
        c=None,
        y=None,
        train=False,
    ):
        return self._forward(x, c=c, y=y, train=train)

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):  
        x, y, c, wsi_id = batch[0], batch[1], batch[2], batch[3]
        return self._forward(x, c=c, y=y, train=False)

    # Concept loss computation
    def _get_concept_loss(self, cpt_state_probs, cpt_labels):
        idx = 0
        bin_losses, cat_losses, cont_losses = [], [], []
        w_bin, w_cat, w_cont = 1, 1, 1
        
        for i, n in enumerate(self.concept_states):
            loss_fn = self.loss_concept_fns[i]
            if n == 0:
                # Continuous concept
                c_pred = cpt_state_probs[:, idx] 
                c_true = cpt_labels[:, i].float()
                concept_loss = loss_fn(c_pred, c_true)
                cont_losses.append(concept_loss)
                idx += 1
            elif n == 1:
                # Binary concept
                c_pred = cpt_state_probs[:, idx]
                c_true = cpt_labels[:, i].float()
                concept_loss = loss_fn(c_pred, c_true)
                bin_losses.append(concept_loss)
                idx += 1
            else:
                # Categorical concept
                c_pred = cpt_state_probs[:, idx:idx + n]
                c_true = cpt_labels[:, i].long()
                concept_loss = loss_fn(c_pred, c_true)
                cat_losses.append(concept_loss)
                idx += n
        
        bin_loss  = torch.mean(torch.stack(bin_losses))  if bin_losses  else torch.tensor(0., device=cpt_state_probs.device)
        cat_loss  = torch.mean(torch.stack(cat_losses))  if cat_losses  else torch.tensor(0., device=cpt_state_probs.device)
        cont_loss = torch.mean(torch.stack(cont_losses)) if cont_losses else torch.tensor(0., device=cpt_state_probs.device)

        total_loss = w_bin*bin_loss + w_cat*cat_loss + w_cont*cont_loss
        return total_loss

    # Main training/validation step
    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
    ):
        x, y, c, wsi_id = batch[0], batch[1], batch[2], batch[3]
        outputs = self._forward(x, c=c, y=y, train=train)
        
        c_sem, c_logits, y_logits, cpt_states, a_w_prob_lst, contexts, concept_attention_weights = outputs

        # Task loss
        if self.task_loss_weight != 0:
            if y_logits is None:
                task_loss = 0
                task_loss_scalar = 0
            else:
                if self.task_type == 'cox':
                    # For Cox regression, y is a tuple of (event, survtime)
                    event, survtime = y
                    hazard_pred = y_logits.reshape(-1)
                    
                    # Debug: Print info on first batch (training or validation)
                    if batch_idx == 0:
                        mode = "TRAIN" if train else "VAL"
                        print(f"\n[COX DEBUG {mode}] Batch {batch_idx}:")
                        print(f"  - event shape: {event.shape}, dtype: {event.dtype}")
                        print(f"  - event stats: min={event.min().item():.2f}, max={event.max().item():.2f}, "
                              f"mean={event.mean().item():.3f}, sum={event.sum().item():.0f} (n_events)")
                        print(f"  - survtime shape: {survtime.shape}, dtype: {survtime.dtype}")
                        print(f"  - survtime stats: min={survtime.min().item():.2f}, max={survtime.max().item():.2f}, "
                              f"mean={survtime.mean().item():.2f}, median={survtime.median().item():.2f}")
                        print(f"  - y_logits shape: {y_logits.shape} -> hazard_pred shape: {hazard_pred.shape}")
                        print(f"  - hazard_pred stats: min={hazard_pred.min().item():.4f}, "
                              f"max={hazard_pred.max().item():.4f}, mean={hazard_pred.mean().item():.4f}, "
                              f"std={hazard_pred.std().item():.4f}")
                        print(f"  - Sample hazard_pred: {hazard_pred[:5].cpu().tolist()}")
                        print(f"  - Sample (event, survtime): {list(zip(event[:5].cpu().tolist(), survtime[:5].cpu().tolist()))}")
                    
                    task_loss = self.loss_task(
                        survtime, event, hazard_pred, y_logits.device
                    )
                    
                    # Debug: Print loss value on first batch
                    if batch_idx == 0:
                        mode = "TRAIN" if train else "VAL"
                        print(f"  - Cox loss: {task_loss.item():.4f}\n")
                else:
                    task_loss = self.loss_task(
                        y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                        y,
                    )
                task_loss_scalar = task_loss.detach()
        else:
            task_loss = 0
            task_loss_scalar = 0

        # Concept loss
        if self.no_concepts:
            # No concepts mode: no concept loss
            loss = task_loss
            concept_loss_scalar = 0.0
        else:
            # Original concept-based mode
            if self.concept_loss_weight != 0:
                concept_loss = self._get_concept_loss(c_sem, c)
                concept_loss_scalar = concept_loss.detach()
                loss = self.concept_loss_weight * concept_loss + task_loss
            else:
                loss = task_loss
                concept_loss_scalar = 0.0

        # Compute single concept metric (F1 for binary/categorical, MSE for continuous)
        if self.no_concepts:
            c_metric_value = 0.0
            c_metric_type = 'f1'
        else:
            # Pass raw logits - compute_concept_metric_single will handle sigmoid/softmax appropriately
            c_metric_value, c_metric_type = compute_concept_metric_single(
                c_sem,  # Raw logits, not sigmoided
                c,
                self.concept_states
            )
        
        # Compute single task metric (F1 for binary/categorical, MSE for regression, C-index for Cox)
        if y_logits is None:
            y_metric_value = 0.0
            y_metric_type = 'f1'
        elif self.task_type == 'cox':
            # C-index computed at epoch end, not per batch
            y_metric_value = 0.0
            y_metric_type = 'c_index'
        elif self.task_type == 'continuous':
            y_metric_value = compute_task_metric_single(
                y_logits,
                y,
                task_type=self.task_type
            )
            y_metric_type = 'mse'
        else:
            y_metric_value = compute_task_metric_single(
                y_logits,
                y,
                task_type=self.task_type
            )
            y_metric_type = 'f1'
        
        # Store metrics based on computed types
        result = {
            "c_f1": c_metric_value if c_metric_type == 'f1' else 0.0,
            "c_mse": c_metric_value if c_metric_type == 'mse' else 0.0,
            "y_f1": y_metric_value if y_metric_type == 'f1' else 0.0,
            "y_mse": y_metric_value if y_metric_type == 'mse' else 0.0,
            "y_c_index": y_metric_value if y_metric_type == 'c_index' else 0.0,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
        }

        return loss, result

    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)
        
        # Log all metrics, but only show concept_loss and task_loss in progress bar
        # Use on_step=True to update progress bar during epoch as batches are processed
        for name, val in result.items():
            # Show only concept_loss and task_loss in progress bar
            if name == "concept_loss" or name == "task_loss":
                self.log(name, val, on_step=True, on_epoch=True, prog_bar=True)
            else:
                self.log(name, val, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        
        # Log all metrics with val_ prefix, show only concept_loss and task_loss in progress bar
        # Use on_epoch=True for validation metrics (typically shown per-epoch)
        for name, val in result.items():
            # Show only concept_loss and task_loss in progress bar
            if name == "concept_loss" or name == "task_loss":
                self.log("val_" + name, val, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log("val_" + name, val, on_step=False, on_epoch=True, prog_bar=False)
        
        result = {
            "val_" + key: val
            for key, val in result.items()
        }
        
        return result

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)
        return result['loss']

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            factor=0.25,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }
