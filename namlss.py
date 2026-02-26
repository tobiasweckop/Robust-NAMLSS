import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from distributions import Distribution

class NAMLSS(nn.Module):

    def __init__(self, formula = None, n_covariates = None, distribution = None, hidden_size = 8):

        # initialize nn.Module
        super(NAMLSS, self).__init__()

        # check, if distribution and formula argument are valid and throw an exception or build the network accordingly
        self.distribution = self._resolve_distribution(distribution)
        self.formula = self._check_formula(formula, n_covariates)
        self.terms = self._parse_formula(self.formula)
        self.module_dict = self._build_modules(self.terms, hidden_size, self.distribution.get_param_count())
    

    def _resolve_distribution(self, distribution):
        if distribution is None:
            raise ValueError("Distribution must be specified.")

        try:
            return Distribution.registry[distribution.lower()]
        except KeyError:
            raise ValueError(f"Distribution '{distribution}' is not available. Please choose from: {list(Distribution.registry.keys())}.")


    def _check_formula(self, formula, n_covariates):
        if formula is None:
            if n_covariates is None:
                raise ValueError("Either 'formula' or 'n_covariates' must be provided.")

            # use n_covariates to generate default formula
            default_formula  = "+".join(str(i) for i in range(n_covariates))

            return default_formula
        

    def _parse_formula(self, formula):

        parsed_terms = []
        terms = formula.split("+")

        for term in terms:
            parts = term.strip().split("*")
            indices = tuple(int(p.strip()) for p in parts)
            parsed_terms.append(indices)

        return parsed_terms


    def _build_modules(self, terms, hidden_size, parameter_count):
        module_dict = nn.ModuleDict()

        for term in terms:
            input_dim = len(term)
            term_key = "*".join(str(i) for i in term)

            module = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, parameter_count)
            )

            module_dict[term_key] = module

        return module_dict


    def forward(self, X):
        # gives each covariate to its corresponding submodule
        # output: list of [observations x parameters] matrices
        component_outputs = [self.module_dict[key](X[:, tuple(int(i) for i in key.split('*'))]) for key in self.module_dict.keys()]

        # [observations x submodules x parameters]
        stacked_array = torch.stack(component_outputs, dim = 1)

        # apply distribution specific transformations to get final parameter vectors
        transformed_parameter_tensor = self.distribution.transform(stacked_array)

        # sums over submodules to get final parameter estimates for each distribution parameter
        parameter_estimate_tensor = torch.sum(transformed_parameter_tensor, dim = 1)

        return parameter_estimate_tensor


    def _prepare_inputs(self, X_train, y_train, X_val=None, y_val=None, starting_weights=None, c=None):

        # Load starting weights if provided
        if starting_weights is not None:
            self.load_state_dict(starting_weights)

        # Ensure c is a tensor
        if c is not None and not torch.is_tensor(c):
            c = torch.tensor(c)

        # Reshape input tensors if necessary
        if X_train.dim() == 1:
            X_train = X_train.unsqueeze(1)
        if y_train.dim() == 2 and y_train.size(1) == 1:
            y_train = y_train.squeeze(1)
        if X_val is not None and X_val.dim() == 1:
            X_val = X_val.unsqueeze(1)
        if y_val is not None:
            if y_val.dim() == 2 and y_val.size(1) == 1:
                y_val = y_val.squeeze(1)

        return X_train, y_train, X_val, y_val, c


    def _snapshot_model_state(self):
        return {key : value.detach().clone() for key, value in self.state_dict().items()}


    def fit(self, X_train, y_train, X_val = None, y_val = None, max_epochs = 10000, lr = 1e-3, weight_decay = 0.0, 
            early_stopping_patience = 10, c = None, starting_weights = None, verbose = False):

        X_train, y_train, X_val, y_val, c = self._prepare_inputs(X_train, y_train, X_val, y_val, starting_weights, c)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):

            # Set model to training mode
            self.train()

            # Forward pass and loss computation
            parameter_tensor = self.forward(X_train)
            train_loss = self.distribution.nll_loss(parameter_tensor, y_train, c)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            val_loss = None
            if X_val is not None and y_val is not None:

                # Set model to evaluation mode to prevent weight updates on validation set
                self.eval()

                with torch.no_grad():
                    validation_parameter_tensor = self.forward(X_val)
                    val_loss = self.distribution.nll_loss(validation_parameter_tensor, y_val, c).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self._snapshot_model_state()
                else:
                    patience_counter += 1

                if (patience_counter >= early_stopping_patience):
                    if verbose:
                        print(f"Validation loss did not improve for {early_stopping_patience} epochs. Early stopping.")
                    self.load_state_dict(best_model_state)
                    break

            if epoch % 100 == 0 or val_loss is not None:
                if verbose:
                    print(f"Epoch {epoch} - Train Loss: {train_loss.item():.4f} - Val Loss: {val_loss:.4f}" if val_loss else f"Epoch {epoch} - Train Loss: {train_loss.item():.4f}")

        return self


    def robust_fit(self, X_train, y_train, X_val, y_val, central_proportion = 0.95, candidate_list = None, max_epochs = 10000):
        
        X_train, y_train, X_val, y_val, c = self._prepare_inputs(X_train, y_train, X_val, y_val)

        if candidate_list is not None:
            candidate_list = candidate_list
        else:
            candidate_list = [None] + [6] + np.round(np.arange(5.1, 2, -0.1),1).tolist() + [1, 0]  # creates list of penalties to test

        best_mse = float("inf")
        state_dict = None

        # TODO: see, if using self instead of a new instance has any disadvantages
        candidate_model = NAMLSS(n_covariates=X_train.shape[1], distribution=self.distribution.__name__)

        for candidate in candidate_list:

            # Fit the model
            candidate_model.fit(X_train, y_train, X_val, y_val, c = candidate)
            parameter_tensor = candidate_model.predict(X_val)

            y_cdf = self.distribution.cdf(parameter_tensor, y_val)
            y_cdf_sorted = torch.sort(y_cdf).values

            # define central quantile interval of interest
            lower_bound = (1 - central_proportion)/2
            upper_bound = 1 - lower_bound

            # keep only quantiles within central interval
            central_mask = (y_cdf_sorted >= lower_bound) & (y_cdf_sorted <= upper_bound)
            central_mask = central_mask.squeeze()
            truncated_y_cdf = y_cdf_sorted[central_mask]

            # compute MSE between empirical and theoretical quantiles in central interval
            expected_quantiles = torch.linspace((1 - central_proportion)/2, 1 - (1 - central_proportion)/2, len(truncated_y_cdf))
            qq_mse = torch.sum((truncated_y_cdf - expected_quantiles)**2) / len(truncated_y_cdf)

            print(f"Candidate c = {candidate}: Truncated QQ MSE = {qq_mse.item():.6f}")

            # save candidate if it improves over current MSE
            if qq_mse < best_mse:
                best_mse = qq_mse
                best_penalty = candidate
                best_state_dict = candidate_model._snapshot_model_state()

        print(f"best penalty identified as c = {best_penalty}")
        # self.fit(X_train, y_train, X_val, y_val, c = best_penalty, starting_weights = best_state_dict, verbose = False) # Note: This was necessary because of a bug with best_state_dict
        # print(f"Model refitted with best penalty c = {best_penalty}.")
        # Hopefully this is not necessary anymore and we can just load the best state dict now.
        # If this creates bugs, just go back to self.fit()
        self.load_state_dict(best_state_dict)
        print(f"Best performing model state loaded.")


    def predict(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(1)

        self.eval()
        with torch.no_grad():
            parameter_tensor = self.forward(X)

        return parameter_tensor
    