import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from distributions import Distribution

class NAMLSS(nn.Module):

    def __init__(self, formula = None, n_covariates = None, distribution = None, numeric_mask = None, global_param_list = None, hidden_size = 8):

        '''
        Initializes the NAMLSS model.

        Arguments:
            formula (str): A string specifying the formula for the model. If None, a default formula will be generated based on n_covariates.
            n_covariates (int): The number of covariates. Required if formula is None.
            distribution (str): The name of the distribution to model. Must be a key in Distribution.registry.
            numeric_mask (torch.BoolTensor): A boolean tensor indicating which covariates are numeric and should be standardized. If None, all covariates are assumed to be numeric.
            global_param_list (list): A list of parameter indices that should be learned as constant. If None, all parameters are learned.
            hidden_size (int): The number of hidden units in each submodule.
        
        '''

        # initialize torch.nn.Module
        super(NAMLSS, self).__init__()

        # validate distribution and formula arguments and set defaults if necessary
        self.distribution = self._resolve_distribution(distribution)
        self.formula = self._check_formula(formula, n_covariates)
        self.terms = self._parse_formula(self.formula)

        # build modules based on the amount of freely learnable parameters
        self.global_param_list = global_param_list or []
        self.global_param_indices = torch.tensor(sorted(self.global_param_list)) - 1
        self.free_param_indices = torch.tensor([i for i in range(self.distribution.get_param_count()) if i not in self.global_param_indices])
        self.free_parameter_count = self._get_free_parameters(self.global_param_indices)
        self.module_dict = self._build_modules(self.terms, hidden_size, self.free_parameter_count)
        self.global_parameter_dict = self._register_global_parameters(self.global_param_indices)
        self.numeric_mask = numeric_mask

        # stores the correct order of parameters loss computation
        self.correct_param_index_tensor = torch.argsort(torch.cat((self.free_param_indices, self.global_param_indices)))

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
        
        else: 
            return formula


    def _parse_formula(self, formula):

        parsed_terms = []
        terms = formula.split("+")

        for term in terms:
            parts = term.strip().split("*")
            indices = tuple(int(p.strip()) for p in parts)
            parsed_terms.append(indices)

        return parsed_terms


    def _build_modules(self, terms, hidden_size, free_parameter_count):
        module_dict = nn.ModuleDict()

        for term in terms:
            input_dim = len(term)
            term_key = "*".join(str(i) for i in term)

            module = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, free_parameter_count)
            )

            module_dict[term_key] = module

        return module_dict


    def _get_free_parameters(self, global_param_indices):

        total_parameter_count = self.distribution.get_param_count()

        # check, if too many constant parameters are provided
        if len(global_param_indices) >= total_parameter_count:
            raise ValueError(f"Number of constant parameters ({len(global_param_indices)}) must be less than total parameters for the distribution ({total_parameter_count}).")
        
        # check, if constant parameter indices are valid
        if any(parameter_index >= total_parameter_count or parameter_index < 0 for parameter_index in global_param_indices):
            raise ValueError(f"Constant parameter indices must be between 0 and {total_parameter_count}.")
        
        # check, if constant parameter indices are unique
        if len(set(global_param_indices)) != len(global_param_indices):
            raise ValueError("Constant parameter indices must be unique.")

        # calculate number of freely learnable parameters
        free_parameter_count = total_parameter_count - len(global_param_indices)

        return free_parameter_count


    def _register_global_parameters(self, global_param_indices):

        param_names = [str(param.item()) for param in global_param_indices]

        # global_node_dict = nn.ParameterDict({param_name: nn.Parameter(torch.zeros(1)) for param_name in param_names})
        global_node_dict = nn.ParameterDict({param_name: nn.Parameter(torch.ones(1)) for param_name in param_names})

        return global_node_dict


    def _prepare_inputs(self, X_train, y_train, X_val = None, y_val = None, starting_weights = None, c = None):

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

        X_train_standardized, X_val_standardized = self._standardize_covariates(X_train, X_val)

        return X_train_standardized, y_train, X_val_standardized, y_val, c


    def _standardize_covariates(self, X_train, X_val = None):

        if self.numeric_mask is None:
            self.numeric_mask = torch.ones(X_train.shape[1], dtype=torch.bool)

        mask = self.numeric_mask

        # Initialize mean/std for all columns
        self.X_mean = torch.zeros(X_train.shape[1], device = X_train.device, dtype = X_train.dtype)
        self.X_std = torch.ones(X_train.shape[1], device = X_train.device, dtype = X_train.dtype)        

        # Compute statistics ONLY on numeric columns
        self.X_mean[mask] = X_train[:, mask].mean(dim=0)
        self.X_std[mask] = X_train[:, mask].std(dim=0) + 1e-8 # add small constant to prevent division by zero

        X_train_standardized = X_train.clone()
        X_train_standardized[:, mask] = (X_train[:, mask] - self.X_mean[mask]) / self.X_std[mask]

        # Standardize X_val and y_val using training statistics
        if X_val is not None:
            X_val_standardized = X_val.clone()
            X_val_standardized[:, mask] = (X_val[:, mask] - self.X_mean[mask]) / self.X_std[mask]
        else:
            X_val_standardized = None

        return X_train_standardized, X_val_standardized


    def _snapshot_model_state(self):
        return {key : value.detach().clone() for key, value in self.state_dict().items()}


    def _assemble_full_parameter_tensor(self, free_parameter_tensor, global_parameter_dict):

        if global_parameter_dict is None or len(global_parameter_dict) == 0:
            return free_parameter_tensor

        # extract parameters from dictionary and stack into tensor
        global_parameter_tensor = torch.cat([p.repeat(free_parameter_tensor.size(0), 1)for p in global_parameter_dict.values()], dim = 1)

        # put free and constant parameters into one tensor
        combined_parameter_tensor = torch.cat((free_parameter_tensor, global_parameter_tensor), dim=1)

        # place columns in correct order for loss computation
        reordered_parameter_tensor = combined_parameter_tensor[:, self.correct_param_index_tensor]

        return reordered_parameter_tensor
    

    def forward(self, X):
        # gives each covariate to its corresponding submodule
        # output: list of [observations x parameters] matrices
        component_outputs = [self.module_dict[key](X[:, tuple(int(i) for i in key.split('*'))]) for key in self.module_dict.keys()]

        full_component_parameter_tensor_list = [self._assemble_full_parameter_tensor(component_output, self.global_parameter_dict) for component_output in component_outputs]

        # [observations x submodules x parameters]
        stacked_array = torch.stack(full_component_parameter_tensor_list, dim = 1)

        # apply distribution specific transformations to get final parameter vectors
        transformed_parameter_tensor = self.distribution.transform(stacked_array)

        # sums over submodules to get final parameter estimates for each distribution parameter
        parameter_estimate_tensor = torch.sum(transformed_parameter_tensor, dim = 1)

        return parameter_estimate_tensor


    def fit(self, X_train, y_train, X_val = None, y_val = None, max_epochs = 10000, lr = 5e-3, weight_decay = 0.0, 
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
                    parameter_validation_tensor = self.forward(X_val)
                    # full_parameter_validation_tensor = self._assemble_full_parameter_tensor(free_parameter_validation_tensor, self.global_parameter_dict)

                    val_loss = self.distribution.nll_loss(parameter_validation_tensor, y_val, c).item()

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


    def robust_fit(self, X_train, y_train, X_val, y_val, central_proportion = 0.95, candidate_list = None, max_epochs = 10000, verbose = False):
        
        X_train, y_train, X_val, y_val, c = self._prepare_inputs(X_train, y_train, X_val, y_val)

        if candidate_list is not None:
            candidate_list = candidate_list
        else:
            candidate_list = [None] + [6] + np.round(np.arange(5.1, 2, -0.1),1).tolist() + [1, 0]  # creates list of penalties to test

        best_mse = float("inf")

        candidate_model = NAMLSS(n_covariates=X_train.shape[1], distribution=self.distribution.__name__, global_param_list = self.global_param_list, hidden_size = 8)

        for candidate in candidate_list:

            # Fit the model
            candidate_model.fit(X_train, y_train, X_val, y_val, c = candidate)
            parameter_tensor = candidate_model.predict_parameters(X_val)

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

            if verbose:
                print(f"Candidate c = {candidate}: Truncated QQ MSE = {qq_mse.item():.6f}")

            # save candidate if it improves over current MSE
            if qq_mse < best_mse:
                best_mse = qq_mse
                best_penalty = candidate
                best_state_dict = candidate_model._snapshot_model_state()

        if verbose:
            print(f"best penalty identified as c = {best_penalty}")

        self.load_state_dict(best_state_dict)

        if verbose:
            print(f"Best performing model state loaded.")


    def predict_parameters(self, X):

        if X.dim() == 1:
            X = X.unsqueeze(1)

        # apply the same scaling as during training
        mask = self.numeric_mask
        X = X.clone()
        X[:, mask] = (X[:, mask] - self.X_mean[mask]) / self.X_std[mask]

        self.eval()
        with torch.no_grad():
            parameter_tensor = self.forward(X)

        return parameter_tensor
    

    def predict_quantiles(self, X, probabilities):

        quantile_list = []

        parameter_tensor = self.predict_parameters(X)

        for i in range(len(probabilities)):
            y_quantiles = self.distribution.icdf(parameter_tensor, torch.tensor(probabilities[i]))
            quantile_list.append(y_quantiles)

        return quantile_list