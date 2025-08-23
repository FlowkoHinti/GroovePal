# top k/ top p
# temperature

# TODO
# If you want to sample tokens from each head simultaneously (e.g., for analysis or ensembling), loop over the heads:
# sampled = {}
# for name, logits in outputs.items():
#     logits_last = logits[:, -1, :]  # (batch, vocab_size)
#     probs = torch.softmax(logits_last, dim=-1)
#     sampled[name] = torch.multinomial(probs, num_samples=1)