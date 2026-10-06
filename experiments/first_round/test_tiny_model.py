"""CPU regression on random tiny GPT-2, without downloading model weights.

This checks computation and state handling, not benchmark performance.
"""
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from common.models.coconut_model import CoconutWrapper
from external.coconut.coconut import Coconut
from experiments.first_round.run import random_edit


class TinyTokenizer:
    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [ord(char) % 29 + 1 for char in text]
        if return_tensors:
            values = torch.tensor([ids])
            return {"input_ids": values, "attention_mask": torch.ones_like(values)}
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        return " ".join(str(int(value)) for value in ids)


def tiny_model():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=37, n_positions=128, n_embd=16, n_layer=2, n_head=2,
                                    resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0,
                                    eos_token_id=36, _attn_implementation="eager"))
    base.eval()
    wrapper = CoconutWrapper()
    wrapper.device = torch.device("cpu")
    wrapper.base_model = base
    wrapper.tokenizer = TinyTokenizer()
    wrapper.start_latent_id, wrapper.end_latent_id, wrapper.latent_token_id, wrapper.eos_token_id = 33, 34, 35, 36
    wrapper.num_latent_placeholders = 3
    wrapper.generation_kwargs = {"max_new_tokens": 5}
    wrapper.coconut_model = Coconut(base, 35, 33, 34, 36)
    return wrapper


class TinyModelTests(unittest.TestCase):
    @torch.no_grad()
    def test_resume_matches_independent_official_forward_and_generation(self):
        model = tiny_model()
        for prompt in ("Hi\n", "A longer question\n"):
            tokens = model._prepare_inputs(prompt)
            reference = model.coconut_model(**tokens, labels=tokens["input_ids"].clone())
            baseline = model.run_baseline(prompt)
            positions = (tokens["input_ids"][0] == 35).nonzero().flatten()
            for step in (1, 2, 3):
                with self.subTest(prompt=prompt, step=step):
                    h, state = model.forward_until_step(prompt, step)
                    before_length = len(state["logits"])
                    before_embeds = state["inputs_embeds"].clone()
                    before_cache = [(k.clone(), v.clone()) for k, v in model._kv_cache_to_legacy_pairs(state["past_key_values"])]
                    expected = reference.inputs_embeds[:, positions, :]
                    torch.testing.assert_close(h, expected[:, step - 1, :])
                    resumed = model.rollout_from_step(h, state)
                    torch.testing.assert_close(resumed["inputs_embeds"][:, positions, :], expected)
                    torch.testing.assert_close(resumed["logits"][:, -1, :], reference.logits[:, -1, :])
                    self.assertEqual(resumed["generated_token_ids"], baseline["generated_token_ids"])
                    self.assertEqual(resumed["continuation"], baseline["continuation"])
                    zero = model.rollout_from_step(torch.zeros_like(h), state)
                    edited = zero["inputs_embeds"][:, positions, :]
                    torch.testing.assert_close(edited[:, :step - 1, :], expected[:, :step - 1, :])
                    self.assertEqual(float(edited[:, step - 1, :].norm()), 0)
                    if step < 3:
                        self.assertGreater(float((edited[:, step:, :] - expected[:, step:, :]).norm()), 0)
                    replay = model.rollout_from_step(h, state)
                    self.assertEqual(replay["generated_token_ids"], resumed["generated_token_ids"])
                    self.assertEqual(len(state["logits"]), before_length)
                    self.assertTrue(torch.equal(state["inputs_embeds"], before_embeds))
                    for (k1, v1), (k2, v2) in zip(before_cache, model._kv_cache_to_legacy_pairs(state["past_key_values"])):
                        self.assertTrue(torch.equal(k1, k2) and torch.equal(v1, v2))

    def test_random_edit_norm_and_seed(self):
        h = torch.arange(1, 17, dtype=torch.float32).reshape(1, -1)
        edited = random_edit(torch, h, 0.5, 42)
        torch.testing.assert_close((edited - h).norm(), 0.5 * h.norm())
        self.assertTrue(torch.equal(edited, random_edit(torch, h, 0.5, 42)))
        self.assertFalse(torch.equal(edited, random_edit(torch, h, 0.5, 43)))

    def test_steps_out_of_range_are_rejected(self):
        model = tiny_model()
        for step in (0, 4):
            with self.assertRaises(ValueError):
                model.forward_until_step("Hi\n", step)


if __name__ == "__main__":
    unittest.main()
