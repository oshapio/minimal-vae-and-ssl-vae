import torch
from torch import nn
import numpy as np

device = "cpu" # TODO: make this proper..

class SupervisedVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        input_size=2,
        hiddens_encode_sizes=None,
        hiddens_classifier=None,
        bottleneck_size=10,
        n_sample_z=3, # TODO: make this functional
        output_activation=nn.Identity(),
        loss="mse",
        kl_term_weight=1.0,
        n_classes=10,
        prob_y=None,
        use_unsupervised_data=False,
    ):
        super(SupervisedVariationalAutoEncoder, self).__init__()

        assert isinstance(
            hiddens_encode_sizes, list
        ), f"`hiddens_encode` needs to be a list but got type `{type(hiddens_encode_sizes)}`"

        encoder_layers = []
        # Add `n_classes` for the one-hot label
        last_size = input_size + n_classes

        ## Encoder q(z | x, y)
        for layer_size in hiddens_encode_sizes:
            encoder_layers.append(nn.Linear(last_size, layer_size))
            encoder_layers.append(nn.ReLU())

            last_size = layer_size
        # Add the bottleneck
        encoder_layers.append(nn.Linear(last_size, bottleneck_size * 2))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []

        last_size = bottleneck_size + n_classes

        ## Decoder p(x | z, y)
        for layer_size in hiddens_encode_sizes[::-1]:
            decoder_layers.append(nn.Linear(last_size, layer_size))
            decoder_layers.append(nn.ReLU())

            last_size = layer_size
        decoder_layers.append(nn.Linear(last_size, input_size * 2))
        # decoder_layers.append(output_activation)

        self.output_activation = output_activation
        self.decoder = nn.Sequential(*decoder_layers)

        ## Classifier q(y|x)
        hiddens_classifier = [300, 100] or hiddens_classifier
        classifier_layers = []

        last_size = input_size
        for layer_size in hiddens_classifier:
            classifier_layers.append(nn.Linear(last_size, layer_size))
            classifier_layers.append(nn.ReLU())

            last_size = layer_size
        classifier_layers.append(nn.Linear(last_size, n_classes))
        classifier_layers.append(nn.Softmax(-1))

        self.classifier = nn.Sequential(*classifier_layers)

        self.bottleneck_size = bottleneck_size
        self.input_size = input_size
        self.n_sample_z = n_sample_z
        self.kl_term_weight = kl_term_weight
        self.loss = loss

        self.bce_loss = nn.BCELoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

        # Assume prior probabilities over classes are known if not given
        self.prob_y = prob_y or torch.tensor([1.0 / n_classes] * n_classes)

        self.use_unsupervised_data = use_unsupervised_data
        self.n_classes = n_classes
        self.class_eye = torch.from_numpy(np.eye(n_classes)).float()

    def encode(self, x, y):
        B = x.shape[0]

        # Get the means and SDs for the approximate posterior network
        q_params = self.encoder(torch.cat((x, y), dim=1))  # .view((B, self.bottleneck_size, 2))
        mu_z, log_sigma_z = q_params[..., : q_params.shape[-1] // 2], q_params[..., q_params.shape[-1] // 2 :]

        # Ensure non-negative variance
        sigma_z = torch.exp(log_sigma_z)

        # Sample from standard Gaussian n_sample_z times => (B, n_samples_z, bottleneck_dim)

        # Because we assume diagonal covariance, sample from B * bottleneck_size * n_sample_z univariate Gaussians

        sampled_eps_z = torch.normal(0.0, 1.0, size=(B, self.bottleneck_size)).to(device)

        # B x n_sample_z x bottleneck_size
        sampled_z = sampled_eps_z * sigma_z + mu_z

        return mu_z, sigma_z, sampled_z

    def decode(self, sampled_z, y):
        B = y.shape[0]

        # (B * n_sample_z) x bottleneck_size
        permuted_flat_sampled_z = torch.cat((sampled_z.view(B, self.bottleneck_size), y), dim=1)

        p_likelihood_params = self.decoder(permuted_flat_sampled_z)  # .view(
        #     (B, self.input_size, 2)
        # )

        mu_x, log_sigma_x = (
            p_likelihood_params[..., : p_likelihood_params.shape[-1] // 2],
            p_likelihood_params[..., p_likelihood_params.shape[-1] // 2 :],
        )
        sigma_x = torch.exp(log_sigma_x)

        mu_x = self.output_activation(mu_x)
        # mu_x.retain_grad()
        # sigma_x.retain_grad()
        # Sample n_sample_x times
        sampled_eps_x = torch.normal(0.0, 1.0, size=(B, self.input_size))#.to(device)

        sampled_x = sampled_eps_x * sigma_x + mu_x

        return mu_x, sigma_x, sampled_x

    def forward(self, x, y):
        """
        x: (n_batches, datapoint_dim)
        y: (pseudo-)label of the data

        returns reconstructed \hatx: (B, n_samples_z, n_samples_x, datapoint_dim)
        and other goodies
        """
        B, _ = x.shape

        mu_z, sigma_z, sampled_z = self.encode(x, y)
        mu_x, sigma_x, sampled_x = self.decode(sampled_z, y)

        return {
            "encoded": sampled_z,
            "mu_z": mu_z,
            "sigma_z": sigma_z,
            "decoded": sampled_x,
            "mu_x": mu_x,
            "sigma_x": sigma_x,
        }

    def get_ELBO(self, input_x, forward_pass_output):
        """
        Returns the evidence lower bound for the (pseudo-)labeled case,
        i.e. L_{labeled} = log p(y) + E_q(z|x, y)[log p(x|z,y)] - D_KL(q(z|x,y)||p(z))
        assuming Gaussians both in forward and backward models _per batch_
        """
        B = input_x.shape[0]

        negative_d_kl_term = 0.5 * torch.sum(
            1.0
            + torch.log(forward_pass_output["sigma_z"] ** 2)
            - forward_pass_output["mu_z"] ** 2
            - forward_pass_output["sigma_z"] ** 2,
            dim=1,
        )  # closed form, simple when diagonal covariance

        if self.loss == "mse":
            likelihood_under_appx_posterior = -torch.sum((input_x - forward_pass_output["mu_x"]) ** 2, dim=1)
        elif self.loss == "cont_bern":
            likelihood_under_appx_posterior = -torch.sum(self.bce_loss(forward_pass_output["mu_x"], input_x), dim=1)

        log_prior_probs = torch.log(self.prob_y[0])

        # For now I assume that p(y) is uniform, so need to index from one-hot
        ELBO = log_prior_probs + (
            likelihood_under_appx_posterior
            if self.kl_term_weight <= 1e-12
            else (self.kl_term_weight * negative_d_kl_term + likelihood_under_appx_posterior)
        )
        return ELBO

    def train_step(self, batch):
        """
        Given a tuple of (input, label data, mask) in a form of tensors
        input dim (B x data_dim), label dim (B x 1), mask (B), does one step of training
        of supervised VAE (M2). Th batch is first divided into two: the ones data
        with labels and without. For labeled data the process is straightword:
        just pass the (x,y) through the net and calculate labeled ELBO. For
        unlabeled data, a classification step is first carried to acquire the
        probabilities q(y|x) and then for each pair (x, y_i), with label y_i
        and corresponding probability q(y_i|x), the labeled ELBO is computed
        and then weighted.
        """

        inputs, labels, labels_flat = batch

        B = inputs.shape[0]

        labeled_data_idxes = torch.where(labels_flat != -1)[0]
        unlabeled_data_idxes = torch.where(labels_flat == -1)[0]

        classification_probs = self.classifier(inputs)

        # Easy part first: labeled data
        lower_bound_labeled, lower_bound_unlabeled = 0.0, 0.0

        if labeled_data_idxes.numel() > 0:
            labeled_data, labeled_idxes = inputs[labeled_data_idxes], labels[labeled_data_idxes]

            labeled_forward_output = self.forward(labeled_data, labeled_idxes)

            ELBO_labeled = self.get_ELBO(labeled_data, labeled_forward_output)

            ce_loss = self.ce_loss(classification_probs[labeled_data_idxes].log(), labels_flat[labeled_data_idxes])
            # torch.sum(torch.log(1e-12 +
            # classification_probs[labeled_data_idxes, labels_flat[labeled_data_idxes]]))

            lower_bound_labeled = (ELBO_labeled - 100.0 * ce_loss).sum() / labeled_data_idxes.numel()
        # Implementation-wise fragile part next: unlabeled data
        if self.use_unsupervised_data and unlabeled_data_idxes.numel() > 0:
            n_unlabel = unlabeled_data_idxes.numel()
            classification_unlabel_probs = classification_probs[unlabeled_data_idxes]

            if True:
                # Slow loop
                stacked_elbos = torch.zeros((n_unlabel, self.n_classes))
                unlabeled_data = inputs[unlabeled_data_idxes]

                for label in range(self.n_classes):
                    one_hot_label = torch.repeat_interleave(self.class_eye[label].unsqueeze(0), n_unlabel, dim=0)
                    pseudolab_forward_output = self.forward(unlabeled_data, one_hot_label)

                    # n_unlabel x 1
                    ELBO_batch_unlabeled = self.get_ELBO(unlabeled_data, pseudolab_forward_output)

                    stacked_elbos[:, label] = ELBO_batch_unlabeled
                log_classification_prob = torch.log(1e-12 + classification_unlabel_probs)

                lower_bound_unlabeled = torch.sum(
                    classification_unlabel_probs * (stacked_elbos - log_classification_prob), dim=1
                ).sum()

            else:
                # fast version
                unlabeled_data = torch.repeat_interleave(
                    inputs[unlabeled_data_idxes].unsqueeze(1), self.n_classes, dim=1
                ).view(n_unlabel * self.n_classes, -1)

                pseudo_labels = torch.repeat_interleave(self.class_eye.unsqueeze(0), n_unlabel, dim=0).view(
                    n_unlabel * self.n_classes, -1
                )

                pseudolab_forward_output = self.forward(unlabeled_data, pseudo_labels)

                # This n_unlabel x n_c
                ELBO_batch_unlabeled = self.get_ELBO(unlabeled_data, pseudolab_forward_output).view(
                    n_unlabel, self.n_classes
                )

                # n_unlabel x n_c
                # probs_repeated = torch.repeat_interleave(classification_unlabel_probs.unsqueeze(0), self.n_classes, dim=0)

                log_classification_prob = torch.log(1e-12 + classification_unlabel_probs)

                lower_bound_unlabeled = (
                    torch.sum(
                        classification_unlabel_probs * (ELBO_batch_unlabeled - log_classification_prob), dim=1
                    ).sum()
                    / n_unlabel
                )

        return {
            "train_loss": -(lower_bound_labeled + lower_bound_unlabeled),
            "classification_probs": classification_probs,
        }
