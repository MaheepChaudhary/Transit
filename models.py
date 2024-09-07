# from huggingface_hub import hf_hub_download
# from transformer_lens.hook_points import HookedRootModule, HookPoint

from imports import *

# torch.autograd.set_detect_anomaly(True)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")



class my_model(nn.Module):
    def __init__(
        self,
        model,
        DEVICE,
        method,
        expansion_factor,
        token_length_allowed,
        layer_intervened,
        intervened_token_idx,
        batch_size,
    ) -> None:
        super(my_model, self).__init__()

        self.model = model
        self.layer_intervened = t.tensor(layer_intervened, dtype=t.int32, device=DEVICE)
        self.intervened_token_idx = t.tensor(
            intervened_token_idx, dtype=t.int32, device=DEVICE
        )
        self.intervened_token_idx = intervened_token_idx
        self.expansion_factor = expansion_factor
        self.token_length_allowed = token_length_allowed
        self.method = method
        self.batch_size = batch_size

        self.DEVICE = DEVICE

        if method == "sae masking openai":
            open_sae_dim = (1, 1, 32768)
            state_dict = t.load(
                f"openai_sae/downloaded_saes/{self.layer_intervened}.pt"
            )
            self.autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict
            )
            self.l4_mask = t.nn.Parameter(
                t.zeros(open_sae_dim, device=DEVICE), requires_grad=True
            )
            for params in self.autoencoder.parameters():
                params.requires_grad = False

        elif method == "vanilla":
            proxy_dim = (1, 1, 1)
            self.proxy = t.nn.Parameter(t.zeros(proxy_dim), requires_grad=True)

    def forward(self, source_ids, base_ids, temperature):
        if self.method == "neuron masking":
            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:
                    vector_source = self.model.transformer.h[
                        self.layer_intervened
                    ].output[0][0]

                with tracer.invoke(base_ids) as runner_:
                    intermediate_output = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    intermediate_output = (1 - l4_mask_sigmoid) * intermediate_output[
                        :, self.intervened_token_idx, :
                    ] + l4_mask_sigmoid * vector_source[:, self.intervened_token_idx, :]
                    assert (
                        intermediate_output.squeeze(1).shape
                        == vector_source[:, self.intervened_token_idx, :].shape
                        == torch.Size([self.batch_size, 768])
                    )
                    self.model.transformer.h[self.layer_intervened].output[0][0][
                        :, self.intervened_token_idx, :
                    ] = intermediate_output.squeeze(1)
                    # self.model.transformer.h[self.layer_intervened].output[0][0][:,self.intervened_token_idx,:] = vector_source[:,self.intervened_token_idx,:]

                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = []
            for index in range(intervened_base_output.shape[0]):
                predicted_text.append(
                    self.model.tokenizer.decode(
                        intervened_base_output[index].argmax(dim=-1)
                    ).split()[-1]
                )

            return intervened_base_output, predicted_text

        

        elif self.method == "sae masking openai":

            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:
                    source = self.model.transformer.h[self.layer_intervened].output[0]

                with tracer.invoke(base_ids) as runner_:

                    base = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    encoded_base, base_info = self.autoencoder.encode(base)
                    encoded_source, source_info = self.autoencoder.encode(source[0])

                    # Clone the tensors to avoid in-place operations
                    encoded_base_modified = encoded_base.clone()
                    encoded_source_modified = encoded_source.clone()

                    assert base_info == source_info

                    # Apply the mask in a non-inplace way
                    modified_base = encoded_base_modified[
                        :, self.intervened_token_idx, :
                    ] * (1 - l4_mask_sigmoid)
                    modified_source = (
                        encoded_source_modified[:, self.intervened_token_idx, :]
                        * l4_mask_sigmoid
                    )

                    # Assign the modified tensors to the correct indices
                    encoded_base_modified = encoded_base_modified.clone()
                    encoded_source_modified = encoded_source_modified.clone()

                    # Combine the modified tensors
                    new_acts = encoded_base_modified.clone()
                    new_acts[:, self.intervened_token_idx, :] = (
                        modified_base + modified_source
                    )

                    iia_vector = self.autoencoder.decode(new_acts, base_info)

                    # Use a copy to avoid in-place modification
                    h_layer_output_copy = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    h_layer_output_copy[:, self.intervened_token_idx, :] = iia_vector[
                        :, self.intervened_token_idx, :
                    ]

                    # Update the model's output with the modified copy
                    self.model.transformer.h[self.layer_intervened].output[0][0][
                        :, :, :
                    ] = h_layer_output_copy

                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = []
            for index in range(intervened_base_output.shape[0]):
                predicted_text.append(
                    self.model.tokenizer.decode(
                        intervened_base_output[index].argmax(dim=-1)
                    ).split()[-1]
                )

            return intervened_base_output, predicted_text




if __name__ == "__main__":
    my_model = my_model()