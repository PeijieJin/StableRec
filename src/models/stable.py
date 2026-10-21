import torch.nn as nn


class StableRecWrapper(nn.Module):
    def __init__(self, base_model, args):
        super().__init__()
        self.base_model = base_model
        self.args = args
        self.item_embedding = base_model.item_embedding
        self.loss_mode = base_model.loss_mode
        self.use_sim = getattr(args, "lambda_sim", 0.0) > 0.0
        self.use_pop = getattr(args, "lambda_pop", 0.0) > 0.0

        def projector():
            return nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(args.hidden_size // 2, 1),
                nn.Sigmoid(),
            )

        self.sim_projector = projector() if self.use_sim else None
        self.pop_projector = projector() if self.use_pop else None

    @property
    def candidate_embedding_weight(self):
        return self.base_model.candidate_embedding_weight

    @property
    def factor_embedding_weight(self):
        return self.base_model.factor_embedding_weight

    def forward(self, input_ids, user_ids=None):
        sequence_output = self.base_model(input_ids, user_ids=user_ids)
        sim_feature = (
            self.sim_projector(sequence_output)
            if self.sim_projector is not None
            else None
        )
        pop_feature = (
            self.pop_projector(sequence_output)
            if self.pop_projector is not None
            else None
        )
        return sequence_output, sim_feature, pop_feature
