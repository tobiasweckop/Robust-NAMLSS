import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import distributions
from distributions import Distribution


class NAMLSS(nn.Module):

    def __init__(self, formula = None, n_covariates = None, distribution = None, numeric_mask = None, global_param_list = None, hidden_size = 16):

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

        # initialize class attributes
        self.c = None
        self.penalty_mse_dict = None

        # initialize torch.nn.Module
        super(NAMLSS, self).__init__()
        self.hidden_size = hidden_size

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

        # stores the correct order of parameters for loss computation
        self.correct_param_index_tensor = torch.argsort(torch.cat((self.free_param_indices, self.global_param_indices)))


    def _resolve_distribution(self, distribution):
        if distribution is None:
            raise ValueError(f"Distribution must be specified. Available distributions: {list(Distribution.registry.keys())}.")

        try:
            return Distribution.registry[distribution.lower()]
        except KeyError:
            raise ValueError(f"Distribution '{distribution}' is not available. Available distributions: {list(Distribution.registry.keys())}.")


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
            raise ValueError(f"Parameter positions must be between 1 and {total_parameter_count}.")
        
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


    def _prepare_inputs(self, X_train, y_train = None, X_val = None, y_val = None, starting_weights = None, c = None):
        ''' 
        Takes raw covariates, gives them the correct shapes and standardizes X.
        '''

        # Load starting weights if provided
        if starting_weights is not None:
            self.load_state_dict(starting_weights)

        # Ensure c is a tensor
        if c is not None and not torch.is_tensor(c):
            c = torch.tensor(c)

        # Reshape input tensors if necessary
        if X_train.dim() == 1:
            X_train = X_train.unsqueeze(1)
        if y_train is not None:
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
        '''
        Takes raw covariates and standardizes them.
        '''

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
    

    def _forward(self, X_standardized):
        '''
        Expects correctly shaped and standardized input X. 
        Returns a tensor of shape [observations x distribution parameters] with the predicted parameters for each observation.
        '''

        # gives each covariate to its corresponding submodule
        # output: list of [observations x parameters] matrices
        component_outputs = [self.module_dict[key](X_standardized[:, tuple(int(i) for i in key.split('*'))]) for key in self.module_dict.keys()]

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
        ''' 
        Expects raw covariates and response. Standardizes X internally before optimizing.
        If validation data are provided, early stopping is used.
        '''

        X_train_standardized, y_train, X_val_standardized, y_val, c = self._prepare_inputs(X_train, y_train, X_val, y_val, starting_weights, c)
        self.chosen_c = c

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):

            # Set model to training mode
            self.train()

            # Forward pass and loss computation
            parameter_tensor = self._forward(X_train_standardized)

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
                    parameter_validation_tensor = self._forward(X_val_standardized)
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

            if epoch % 100 == 0 and verbose:
                print(f"Epoch {epoch} - Train Loss: {train_loss.item():.4f} - Val Loss: {val_loss:.4f}" if val_loss is not None else f"Epoch {epoch} - Train Loss: {train_loss.item():.4f}")

        return self


    def fit_batches(self, X_train, y_train, X_val=None, y_val=None, max_epochs=10000, lr=5e-3, weight_decay=0.0, 
                    early_stopping_patience=10, c=None, starting_weights=None, verbose=False, batch_size=None, shuffle=True):
        '''
        Expects raw covariates and response. Standardizes X internally before optimizing.
        If validation data are provided, early stopping is used.

        batch_size:
            None -> full-batch training, without DataLoader
            integer -> mini-batch training using DataLoader

        shuffle:
            Only relevant for mini-batch training.
        '''

        # Prepare inputs
        X_train_standardized, y_train, X_val_standardized, y_val, c = (self._prepare_inputs(X_train, y_train, X_val, y_val, starting_weights, c))
        self.chosen_c = c

        # Set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Create batches
        if batch_size is not None:

            train_dataset = torch.utils.data.TensorDataset(X_train_standardized, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            train_dataset_size = len(train_dataset)

        else:
            train_loader = None
            train_dataset_size = X_train_standardized.shape[0]

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None


        for epoch in range(max_epochs):

            self.train()

            # Full batch training
            if train_loader is None:

                parameter_tensor = self._forward(X_train_standardized)
                train_loss = self.distribution.nll_loss(parameter_tensor, y_train, c)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_loss_value = train_loss.item()

            # Mini batch training
            else:
                epoch_loss = 0.0

                for X_batch, y_batch in train_loader:

                    parameter_tensor = self._forward(X_batch)
                    batch_loss = self.distribution.nll_loss(parameter_tensor, y_batch, c)

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    epoch_loss += (batch_loss.item() * X_batch.shape[0])

                train_loss_value = (epoch_loss / train_dataset_size)


            val_loss = None

            if X_val_standardized is not None and y_val is not None:

                self.eval()

                with torch.no_grad():

                    parameter_validation_tensor = self._forward(X_val_standardized)

                    val_loss = self.distribution.nll_loss(parameter_validation_tensor, y_val, c).item()


                if val_loss < best_val_loss:

                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self._snapshot_model_state()

                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:

                    if verbose:
                        print(f"Validation loss did not improve for {early_stopping_patience} epochs. Early stopping at epoch {epoch}.")

                    self.load_state_dict(best_model_state)
                    break


            if epoch % 100 == 0 and verbose:
                if val_loss is not None:

                    print(f"Epoch {epoch} - Train Loss: {train_loss_value:.4f} - Val Loss: {val_loss:.4f}")

                else:
                    print(f"Epoch {epoch} - Train Loss: {train_loss_value:.4f}")

        return self
    

    def robust_fit(self, X_train, y_train, X_val, y_val, central_proportion = 0.95, penalty_list = None, max_epochs = 10000, verbose = False, batch_size = None, plot = False):

        if y_train.ndim == 2:
            assert y_train.shape[1] == 1
            y_train = y_train.squeeze(1)

        if y_val.ndim == 2:
            assert y_val.shape[1] == 1
            y_val = y_val.squeeze(1)

        if penalty_list is not None:
            penalty_list = penalty_list
        else:
            penalty_list = [None] + np.round(np.arange(7.0, 1, -0.1),1).tolist()  # creates list of penalties to test

        best_mse = float("inf")
        self.penalty_mse_dict = {}

        candidate_model = NAMLSS(n_covariates=X_train.shape[1], distribution=self.distribution.__name__, global_param_list = self.global_param_list, hidden_size = self.hidden_size)

        for penalty_candidate in penalty_list:

            # Fit the model
            candidate_model.fit_batches(X_train, y_train, X_val, y_val, c = penalty_candidate, max_epochs = max_epochs, batch_size = batch_size)
            # candidate_model.fit_batches(X_train, y_train, X_val, y_val, c = penalty_candidate, max_epochs = max_epochs, batch_size = batch_size)

            # Predict parameters based on validation data
            parameter_tensor = candidate_model.predict_parameters(X_val)

            # Calculate cdf-values of y given estimated parameters
            y_cdf = self.distribution.cdf(parameter_tensor, y_val)
            y_cdf_sorted = torch.sort(y_cdf).values

            # define central quantile interval of interest
            lower_bound = (1 - central_proportion)/2
            upper_bound = 1 - lower_bound

            # keep only quantiles within central interval
            central_mask = (y_cdf_sorted >= lower_bound) & (y_cdf_sorted <= upper_bound)
            central_mask = central_mask
            truncated_y_cdf = y_cdf_sorted[central_mask]

            # compute MSE between empirical and theoretical quantiles in central interval
            expected_quantiles = torch.linspace((1 - central_proportion)/2, 1 - (1 - central_proportion)/2, len(truncated_y_cdf), device = truncated_y_cdf.device)
            qq_mse = torch.sum((truncated_y_cdf - expected_quantiles)**2) / len(truncated_y_cdf)

            if penalty_candidate is None: 
                penalty_candidate_name = "No penalty"
            else:
                penalty_candidate_name = penalty_candidate

            self.penalty_mse_dict[penalty_candidate_name] = qq_mse.item()

            if verbose:
                print(f"Candidate c = {penalty_candidate}: Truncated QQ MSE = {qq_mse.item():.6f}")

            # save candidate if it improves over current MSE
            if qq_mse < best_mse:
                best_mse = qq_mse
                best_penalty = penalty_candidate
                best_state_dict = candidate_model._snapshot_model_state()
                self.X_mean = candidate_model.X_mean
                self.X_std = candidate_model.X_std
                self.chosen_c = penalty_candidate


        if verbose:
            print(f"best penalty identified as c = {best_penalty}")

        self.load_state_dict(best_state_dict)

        if verbose:
            print(f"Best performing model state loaded.")

        if plot:
            self.plot_PIT(X_val, y_val, central_proportion)


    def robust_fit_grid(self, X_train, y_train, X_val, y_val, central_proportion_list=None, penalty_list=None, max_epochs=10000, verbose=False, batch_size=None, plot=False):

        ''' 
        Evaluates PIT performance for a list of robustness penalties across multiple central_proportion values.
        Calculates relative regret for each c across central proportions and selects the c
        with the lowest median relative regret as the final penalty.
        '''

        if y_train.ndim == 2:
            assert y_train.shape[1] == 1
            y_train = y_train.squeeze(1)

        if y_val.ndim == 2:
            assert y_val.shape[1] == 1
            y_val = y_val.squeeze(1)

        if central_proportion_list is None:
            central_proportion_list = [0.80, 0.90, 0.95]

        central_proportion_list = sorted(central_proportion_list)

        if penalty_list is not None:
            penalty_list = penalty_list
        else:
            penalty_list = [None] + np.round(np.arange(7.0, 0, -0.1), 1).tolist()

        self.penalty_mse_dict = {}
        self.best_c_by_central_proportion = {}
        self.best_mse_by_central_proportion = {}

        for central_proportion in central_proportion_list:

            self.penalty_mse_dict[central_proportion] = {}

            best_mse = float("inf")
            best_penalty = None

            if verbose:
                print(f"\n===== Central proportion = {central_proportion:.2f} =====")

            candidate_model = NAMLSS(n_covariates=X_train.shape[1], distribution=self.distribution.__name__, global_param_list=self.global_param_list, hidden_size=self.hidden_size)

            for penalty_candidate in penalty_list:

                candidate_model.fit_batches(X_train, y_train, X_val, y_val, c=penalty_candidate, max_epochs=max_epochs, batch_size=batch_size)

                parameter_tensor = candidate_model.predict_parameters(X_val)

                y_cdf = self.distribution.cdf(parameter_tensor, y_val)
                y_cdf_sorted = torch.sort(y_cdf).values

                lower_bound = (1 - central_proportion) / 2
                upper_bound = 1 - lower_bound

                central_mask = (y_cdf_sorted >= lower_bound) & (y_cdf_sorted <= upper_bound)
                truncated_y_cdf = y_cdf_sorted[central_mask]

                expected_quantiles = torch.linspace(lower_bound, upper_bound, len(truncated_y_cdf), device=truncated_y_cdf.device)
                qq_mse = torch.mean((truncated_y_cdf - expected_quantiles) ** 2)

                penalty_candidate_name = "No penalty" if penalty_candidate is None else penalty_candidate

                self.penalty_mse_dict[central_proportion][penalty_candidate_name] = qq_mse.item()

                if verbose:
                    print(f"Candidate c = {penalty_candidate}: Truncated QQ MSE = {qq_mse.item():.6f}")

                if qq_mse.item() < best_mse:
                    best_mse = qq_mse.item()
                    best_penalty = penalty_candidate

            self.best_c_by_central_proportion[central_proportion] = best_penalty
            self.best_mse_by_central_proportion[central_proportion] = best_mse

            if verbose:
                print(f"Best c for central proportion {central_proportion:.2f}: {best_penalty}")

        self.central_proportion_results = []

        for central_proportion in central_proportion_list:
            self.central_proportion_results.append({"central_proportion": central_proportion, "best_c": self.best_c_by_central_proportion[central_proportion], "best_qq_mse": self.best_mse_by_central_proportion[central_proportion]})

        if verbose:
            print("\n===== Summary =====")
            for central_proportion in central_proportion_list:
                print(f"Central proportion = {central_proportion:.2f} | Best c = {self.best_c_by_central_proportion[central_proportion]} | QQ-MSE = {self.best_mse_by_central_proportion[central_proportion]:.6f}")


        ########## Calculate relative regret ##########

        common_c_values = None
        regret_matrix = []

        for central_proportion in central_proportion_list:

            c_values = []
            mse_values = []

            for penalty_name, mse in self.penalty_mse_dict[central_proportion].items():

                if penalty_name == "No penalty":
                    continue

                c_values.append(float(penalty_name))
                mse_values.append(mse)

            order = np.argsort(c_values)

            c_values = np.array(c_values)[order]
            mse_values = np.array(mse_values)[order]

            if common_c_values is None:
                common_c_values = c_values
            else:
                assert np.array_equal(common_c_values, c_values), "Penalty grids differ between central proportions."

            minimum_mse = np.min(mse_values)

            relative_regret = mse_values / minimum_mse

            regret_matrix.append(relative_regret)

        regret_matrix = np.array(regret_matrix)

        self.relative_regret_matrix = regret_matrix
        self.relative_regret_c_values = common_c_values

        self.mean_relative_regret = np.mean(regret_matrix, axis=0)
        self.median_relative_regret = np.median(regret_matrix, axis=0)

        best_mean_c = common_c_values[np.argmin(self.mean_relative_regret)]
        best_median_c = common_c_values[np.argmin(self.median_relative_regret)]

        self.best_mean_regret_c = best_mean_c
        self.best_median_regret_c = best_median_c

        ########## Choose final c using lowest median relative regret ##########

        final_c = best_median_c
        self.chosen_c = final_c

        if verbose:
            print("\n===== Relative regret summary =====")
            print(f"Best c by mean relative regret: c = {best_mean_c:.1f}")
            print(f"Best c by median relative regret: c = {best_median_c:.1f}")
            print(f"Final selected c: c = {final_c:.1f}")


        ########## Fit final model using selected c ##########

        final_model = NAMLSS(n_covariates=X_train.shape[1], distribution=self.distribution.__name__, global_param_list=self.global_param_list, hidden_size=self.hidden_size)

        final_model.fit_batches(X_train, y_train, X_val, y_val, c=final_c, max_epochs=max_epochs, batch_size=batch_size)

        best_state_dict = final_model._snapshot_model_state()

        self.X_mean = final_model.X_mean
        self.X_std = final_model.X_std

        self.load_state_dict(best_state_dict)

        if verbose:
            print(f"\nFinal c = {self.chosen_c}")
            print("Final model state loaded.")


        ########## Plot results ##########

        if plot:

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(10, 10))


            ########## Plot 1: Raw QQ-MSE ##########

            for central_proportion in central_proportion_list:

                c_values = []
                mse_values = []

                for penalty_name, mse in self.penalty_mse_dict[central_proportion].items():

                    if penalty_name == "No penalty":
                        continue

                    c_values.append(float(penalty_name))
                    mse_values.append(mse)

                order = np.argsort(c_values)

                c_values = np.array(c_values)[order]
                mse_values = np.array(mse_values)[order]

                axes[0].plot(c_values, mse_values, label=f"Central proportion = {central_proportion:.2f}")

                best_c = self.best_c_by_central_proportion[central_proportion]

                if best_c is not None:
                    axes[0].scatter(best_c, self.best_mse_by_central_proportion[central_proportion], s=80)

            axes[0].set_xlabel("Penalty c")
            axes[0].set_ylabel("QQ-MSE")
            axes[0].set_title("QQ-MSE across penalty values and central proportions")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)


            ########## Plot 2: Relative regret ##########

            for i, central_proportion in enumerate(central_proportion_list):
                axes[1].plot(common_c_values, regret_matrix[i], label=f"Central proportion = {central_proportion:.2f}")

            axes[1].plot(common_c_values, self.mean_relative_regret, color="black", linewidth=3, label="Mean relative regret")
            axes[1].plot(common_c_values, self.median_relative_regret, color="black", linestyle="--", linewidth=2, label="Median relative regret")

            axes[1].axvline(best_mean_c, color="black", linestyle=":", linewidth=2, label=f"Best mean regret: c = {best_mean_c:.1f}")
            axes[1].axvline(best_median_c, color="gray", linestyle=":", linewidth=2, label=f"Best median regret: c = {best_median_c:.1f}")

            axes[1].set_xlabel("Penalty c")
            axes[1].set_ylabel("Relative QQ-MSE")
            axes[1].set_title("Relative QQ-MSE across central proportions")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()


    def predict_parameters(self, X):

        if X.dim() == 1:
            X = X.unsqueeze(0)

        X_standardized = (X - self.X_mean)/self.X_std

        with torch.no_grad():
            parameter_tensor = self._forward(X_standardized = X_standardized)

        return parameter_tensor


    def marginal_effects(self, X):

        if X.dim() == 1:
            X = X.unsqueeze(0)

        X_standardized = (X - self.X_mean)/self.X_std

        with torch.no_grad():
            component_outputs = [self.module_dict[key](X_standardized[:, tuple(int(i) for i in key.split('*'))]) for key in self.module_dict.keys()]
            full_component_parameter_tensor_list = [self._assemble_full_parameter_tensor(component_output, self.global_parameter_dict) for component_output in component_outputs]

            # [observations x submodules x parameters]
            stacked_array = torch.stack(full_component_parameter_tensor_list, dim = 1)

            # apply distribution specific transformations to get final parameter vectors
            transformed_parameter_tensor = self.distribution.transform(stacked_array)

        return transformed_parameter_tensor


    def plot_marginal_effects(self, X, covariate_indices = None, feature_names = None, parameter_names = None, show=True):

        # Ensure correct input shape
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # Get marginal effects
        marginal_tensor = self.marginal_effects(X)

        # Convert to NumPy
        marginal_np = marginal_tensor.detach().cpu().numpy()

        n_observations, n_terms, n_parameters = marginal_np.shape

        # Select terms to plot
        if covariate_indices is None:
            covariate_indices = list(range(n_terms))
        else:
            covariate_indices = list(covariate_indices)

        # Check term indices
        invalid_indices = [i for i in covariate_indices if i < 0 or i >= n_terms]

        if len(invalid_indices) > 0:
            raise ValueError(f"Invalid term indices {invalid_indices}. Valid term indices are 0 to {n_terms - 1}.")

        if len(covariate_indices) == 0:
            raise ValueError("variables_to_plot must contain at least one term.")

        # Parameter names
        if parameter_names is None:
            if self.distribution == distributions.Normal and n_parameters == 2:
                parameter_names = [r"$\mu$", r"$\sigma$"]
            else:
                parameter_names = [f"Parameter {i + 1}" for i in range(n_parameters)]

        if len(parameter_names) != n_parameters:
            raise ValueError(f"parameter_names contains {len(parameter_names)} names, but the model has {n_parameters} distribution parameters.")


        marginal_np = marginal_np - marginal_np.mean(axis=0, keepdims=True)

        # Determine common y-axis limits for each parameter
        y_limits = []

        for parameter_idx in range(n_parameters):
            values = marginal_np[:, covariate_indices, parameter_idx]
            max_abs = np.max(np.abs(values))
            if max_abs == 0:
                max_abs = 1.0
            y_limits.append(max_abs * 1.05)


        figsize = (5 * len(covariate_indices), 3.2 * n_parameters)

        fig, axes = plt.subplots(n_parameters, len(covariate_indices), figsize=figsize, squeeze=False)

        # Plot each term
        for plot_column, term_idx in enumerate(covariate_indices):

            term = self.terms[term_idx]
            covariate_matrix = X[:, term]

            # Determine x-axis
            if len(term) == 1:
                x_values = covariate_matrix.squeeze(1).detach().cpu().numpy()
                sorting_indices = np.argsort(x_values)
                x_sorted = x_values[sorting_indices]
            else:
                sorting_indices = np.arange(len(X))
                x_sorted = np.arange(len(X))

            # Determine feature name
            if feature_names is None:
                if len(term) == 1:
                    feature_name = f"X{term[0]}"
                else:
                    feature_name = " × ".join(f"X{i}" for i in term)
            else:
                feature_name = " × ".join(feature_names[i] for i in term)

            # Plot each distribution parameter
            for parameter_idx in range(n_parameters):

                ax = axes[parameter_idx, plot_column]

                y_values = marginal_np[:, term_idx, parameter_idx]
                y_sorted = y_values[sorting_indices]

                ax.plot(x_sorted, y_sorted, linewidth=2.5)

                ax.set_xlabel(feature_name, fontsize=14)

                if plot_column == 0:
                    ax.set_ylabel("Centered Contribution", fontsize=14)
                else:
                    ax.set_ylabel("")

                ax.set_ylim(-y_limits[parameter_idx], y_limits[parameter_idx])
                ax.tick_params(axis="both", labelsize=14 - 2)
                ax.grid(alpha=0.3)

        # Add parameter labels on the left
        for parameter_idx, parameter_name in enumerate(parameter_names):
            axes[parameter_idx, 0].text(-0.25, 0.5, parameter_name, transform=axes[parameter_idx, 0].transAxes, ha="center", va="center", fontsize= 14 + 6)

        # Layout
        fig.tight_layout()

        # Display figure
        if show:
            plt.show()

        return fig, axes


    def predict_quantiles(self, X, probabilities):

        ''' Takes an unstandardized version of X and predicts quantiles for the specified probabilities.'''

        quantile_list = []

        if X.dim() == 1:
            X = X.unsqueeze(0)
            
        X_standardized = (X - self.X_mean)/self.X_std

        with torch.no_grad():
            parameter_tensor = self._forward(X_standardized)

            for i in range(len(probabilities)):

                current_probability = torch.tensor(probabilities[i]).repeat(parameter_tensor.shape[0])

                y_quantiles = self.distribution.icdf(parameter_tensor, torch.as_tensor(current_probability))
                quantile_list.append(y_quantiles)

        return quantile_list
    

    def plot_PIT(self, X, y, central_proportion = None):

        if X.dim() == 1:
            X = X.unsqueeze(0)

        if y.ndim == 2:
            assert y.shape[1] == 1
            y = y.squeeze(1)

        if central_proportion is None:
            central_proportion = 1

        predicted_parameter_tensor = self.predict_parameters(X)

        # calculating cdf values for each observation
        y_cdf = self.distribution.cdf(predicted_parameter_tensor, y)
        y_cdf_sorted = torch.sort(y_cdf).values

        # define central quantile interval of interest
        lower_bound = (1 - central_proportion)/2
        upper_bound = 1 - lower_bound

        # keep only quantiles within central interval
        central_mask = (y_cdf_sorted >= lower_bound) & (y_cdf_sorted <= upper_bound)
        central_mask = central_mask
        truncated_y_cdf = y_cdf_sorted[central_mask]

        fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)

        # Plot the histogram
        counts, bins, _ = ax.hist(truncated_y_cdf, bins = 50, density = True, alpha = 0.7, edgecolor = "black", linewidth = 0.5)

        if central_proportion < 1:
            # Add vertical lines to mark central interval
            ax.axvline(lower_bound, color = "tab:red", linewidth = 1.5, label = "Central interval")
            ax.axvline(upper_bound, color = "tab:red", linewidth = 1.5)

        # Calculate expected uniform density over the central interval
        expected_density = 1 / (upper_bound - lower_bound)

        # Calculate quantile MSE
        expected_quantiles = torch.linspace((1 - central_proportion)/2, 1 - (1 - central_proportion)/2, len(truncated_y_cdf), device = truncated_y_cdf.device)
        qq_mse = torch.mean((truncated_y_cdf - expected_quantiles)**2)

        # Add horizontal line to show expected bar height inside central interval
        ax.hlines(expected_density, xmin = lower_bound, xmax = upper_bound, color="black", linestyle = "--",  linewidth=1.5, label="Uniform PIT Density")
        ax.set_ylim((0, 4 * expected_density))

        ax.set_xlabel("PIT value", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(rf"PIT histogram ($\mathrm{{MSE}} = {qq_mse.item():.2e}$)", fontsize = 11)
        ax.tick_params(labelsize=10)
        ax.legend(frameon=False, fontsize=9)

        fig.tight_layout()
        plt.show()


    def plot_normal_residuals(self, X, y_observed, central_proportion = 0.95, title = None):

        if self.distribution != distributions.Normal:
            raise ValueError(f"This diagnostic is only available for Normal models, but the current distribution is {self.distribution}.")

        if y_observed.ndim == 2:
            assert y_observed.shape[1] == 1
            y_observed = y_observed.squeeze(1)

        # Get predicted distribution parameters
        parameter_tensor = self.predict_parameters(X)

        mu = parameter_tensor[:, 0]
        sigma = parameter_tensor[:, 1]

        standard_normal_tensor = torch.tensor([[0.0, 1.0]])
        std_residuals = (y_observed - mu) / sigma

        p_low = torch.tensor([(1 - central_proportion) / 2])
        p_high = torch.tensor([(1 + central_proportion) / 2])

        # Correct ordering
        lower_bound = distributions.Normal.icdf(parameter_tensor = standard_normal_tensor, p = p_low).item()
        upper_bound = distributions.Normal.icdf(parameter_tensor = standard_normal_tensor, p = p_high).item()

        # Histogram requires NumPy
        residuals_np = std_residuals

        bar_heights, bin_edges = np.histogram(residuals_np, bins = 100, density = False)

        bin_widths = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + bin_widths / 2

        central_bins = (bin_centers >= lower_bound) & (bin_centers <= upper_bound)

        area_in_range = np.sum(bar_heights[central_bins] * bin_widths[central_bins])

        rescaled_bars = bar_heights * central_proportion / area_in_range

        plt.bar(bin_centers, rescaled_bars, width = bin_widths, alpha = 0.6, label = "Standardized Residuals")

        x = torch.linspace(-4, 4, 500)
        parameters = standard_normal_tensor.repeat(len(x), 1)
        pdf = distributions.Normal.pdf(parameters, x)

        plt.plot(x, pdf, "--", lw = 2, color = "red", label = "Standard Normal Density")

        plt.xlabel("Residuals")
        plt.ylabel("Rescaled Density")
        plt.xlim((-6, 6))
        plt.ylim((0, 0.6))
        plt.title(title if title is not None else "Standardized Residual Plot")
        plt.legend()
        plt.tight_layout()
        plt.show()

