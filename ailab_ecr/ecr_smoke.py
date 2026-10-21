"""ECR smoke test: one recommender training step with tiny random models, no data download.

Checks that ECR's own model code (model_gpt2.py, model_prompt.py, config.py) runs on the
container's library versions and GPU, and that the seed behaves as expected:
same seed -> identical result, different seed -> different result.

Usage:  python ecr_smoke.py /path/to/ECR/src_emo
"""
import sys
import torch

# PATCH 1: ECR's config.py calls torch.set_deterministic(), which newer PyTorch renamed.
# warn_only=True: some GPU graph ops have no deterministic version; warn instead of crashing.
torch.set_deterministic = lambda flag: torch.use_deterministic_algorithms(flag, warn_only=True)

sys.path.insert(0, sys.argv[1])
from transformers import GPT2Config, RobertaConfig, RobertaModel, AdamW  # noqa: E402
from model_gpt2 import PromptGPT2forCRS  # noqa: E402
from model_prompt import KGPrompt  # noqa: E402
from config import Emo_List, seed_torch  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"

try:  # ECR stores the knowledge graph as a torch_sparse SparseTensor (edge types as values)
    from torch_sparse import SparseTensor
except ImportError:  # PATCH 2 (only without torch_sparse): ECR's own commented-out line 133
    SparseTensor = None

    def _get_entity_embeds(self):
        e = self.kg_encoder(self.node_embeds, self.edge_index, self.edge_type) + self.node_embeds
        e = self.entity_proj1(e) + e
        return self.entity_proj2(e)

    KGPrompt.get_entity_embeds = _get_entity_embeds


def one_step(seed):
    seed_torch(seed)
    model = PromptGPT2forCRS(GPT2Config(n_embd=64, n_layer=2, n_head=4, vocab_size=200,
                                        n_positions=128, pad_token_id=0)).to(dev).requires_grad_(False)
    text_encoder = RobertaModel(RobertaConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
                                              intermediate_size=64, vocab_size=200)).to(dev).requires_grad_(False)
    n_entity, n_rel = 50, 3
    g = torch.Generator().manual_seed(123)  # the graph is data: fixed, independent of the training seed
    edge_index = torch.randint(0, n_entity, (2, 200), generator=g).to(dev)
    edge_type = torch.randint(0, n_rel, (200,), generator=g).to(dev)
    adj = edge_index if SparseTensor is None else SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_type)
    prompt_encoder = KGPrompt(64, 64, 4, 2, 2, n_entity=n_entity, num_relations=n_rel, num_bases=2,
                              edge_index=adj, edge_type=edge_type, n_prefix_rec=10,
                              n_emotion=len(Emo_List)).to(dev)
    opt = AdamW([p for p in prompt_encoder.parameters() if p.requires_grad], lr=1e-3)

    bs = 4
    token_embeds = text_encoder(input_ids=torch.randint(0, 200, (bs, 12), device=dev)).last_hidden_state
    prompt_embeds, _ = prompt_encoder(
        entity_ids=torch.randint(0, n_entity, (bs, 5), device=dev), token_embeds=token_embeds,
        emotion_ids=torch.randint(0, len(Emo_List), (bs, 5, 2), device=dev),
        emotion_probs=torch.rand(bs, 5, 2, device=dev), output_entity=True, use_rec_prefix=True)
    out = model(input_ids=torch.randint(0, 200, (bs, 16), device=dev), prompt_embeds=prompt_embeds,
                entity_embeds=prompt_encoder.get_entity_embeds(), rec=True,
                rec_labels=torch.randint(0, n_entity, (bs,), device=dev))
    out.rec_loss.backward()
    opt.step()
    return round(float(out.rec_loss), 6), torch.topk(out.rec_logits, k=5, dim=-1).indices[0].tolist()


if __name__ == "__main__":
    print("torch", torch.__version__, "| device", dev, "| torch_sparse", SparseTensor is not None)
    a, b, c = one_step(0), one_step(0), one_step(1)
    print("seed 0, run 1 (loss, top-5 items):", a)
    print("seed 0, run 2 (loss, top-5 items):", b)
    print("seed 1        (loss, top-5 items):", c)
    print("same seed reproducible:", a == b, "| different seed differs:", a != c)
