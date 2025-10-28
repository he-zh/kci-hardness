import torch
import torch.optim as optim
from kernels import LinearModel, RBFModel
from criterion import LeaveOneOutCriterion
from u_estimator import compute_hsic
from pval_computations import compute_pval
from utils import solve_regularized_kernel_matrix_system


MODEL_CLASSES = {
    "linear": LinearModel,
    "rbf": RBFModel,
}

class KernelSelection:
    def __init__(self, model_a='linear', model_b='linear',
                 model_c='rbf', model_ca='rbf', model_cb='rbf',
                 epochs=200, lr=0.01, input_dim=10, hidden_dim=20,
                 gamma_c=1.0, gamma_ca=1.0, gamma_cb=1.0, ridge_lambda=0.01,
                 channel_c=0, channel_ca=0, channel_cb=0,
                 is_trainable_c=False, is_trainable_ca=True, is_trainable_cb=True, 
                 early_stopping=False, verbose=False,
                 pval_approx_type='wild', n_wild_bootstrap_samples=1000,
                 device='cpu',
                 **kwargs):
        self.epochs = epochs
        self.lr = lr
        self.is_hsic_baised = pval_approx_type == 'gamma'
        self.pval_approx_type = pval_approx_type
        self.n_wild_bootstrap_samples = n_wild_bootstrap_samples
        self.device = device
        # Initialize kernel parameters/models
        self.kernel_a = self._initialize_model(model_type=model_a, gamma=1.0, is_trainable=False)
        self.kernel_b = self._initialize_model(model_type=model_b, gamma=1.0, is_trainable=False)
        self.kernel_c = self._initialize_model(model_type=model_c, input_dim=input_dim, 
                                               hidden_dim=hidden_dim, gamma=gamma_c, is_trainable=is_trainable_c, channel=channel_c,
                                               **kwargs)
        self.kernel_ca = self._initialize_model(model_type=model_ca, input_dim=input_dim, 
                                                hidden_dim=hidden_dim, gamma=gamma_ca, ridge_lambda=ridge_lambda, channel=channel_ca,
                                                is_trainable=is_trainable_ca, **kwargs)
        self.kernel_cb = self._initialize_model(model_type=model_cb, input_dim=input_dim,
                                                hidden_dim=hidden_dim, gamma=gamma_cb, ridge_lambda=ridge_lambda, channel=channel_cb,
                                                is_trainable=is_trainable_cb, **kwargs)
        self.early_stopping = early_stopping
        self.verbose = verbose

    def _initialize_model(self, model_type='base', input_dim=10, hidden_dim=20, gamma=1.0, ridge_lambda=0.1,
                          is_trainable=False, channel=0, **kwargs):
        model_class = MODEL_CLASSES.get(model_type)
        if model_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_class(input_dim=input_dim, hidden_dim=hidden_dim, gamma=gamma, ridge_lambda=ridge_lambda,
                           is_trainable=is_trainable, channel=channel, **kwargs).to(self.device)


    def _select_kernel_parameters(self, X, K_YY, model_x):
        criterion = LeaveOneOutCriterion()
        kernel_params = []; ridge_lambda_params = []
        for name, param in model_x.named_parameters():
            if 'lambda' not in name:
                kernel_params.append(param)
            else:
                ridge_lambda_params.append(param)
        optimizer = optim.Adam(kernel_params, lr=self.lr)
        optimizer_ridge_lambda = optim.Adam(ridge_lambda_params, lr=0.001)

        model_x.train()
        if self.early_stopping:
            best_loss = float('inf')
            patience = 50
            wait = 0  # Counter for epochs since last improvement

        for epoch in range(self.epochs):    
            optimizer.zero_grad()
            optimizer_ridge_lambda.zero_grad()
            K_XX = model_x(X)
            loss = criterion(K_XX, K_YY, model_x.ridge_lambda)
            loss.backward()
            optimizer.step()
            optimizer_ridge_lambda.step()
            model_x.ridge_lambda.data.clamp_(min=1e-5)
            # Check for early stopping
            if self.early_stopping:
                current_loss = loss.item()
                if current_loss < best_loss - 1e-8:  # Small tolerance to avoid float precision issues
                    best_loss = current_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
            if (epoch) % 50 == 0 and self.verbose:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4e}, Ridge: {model_x.ridge_lambda.item():.4e}, Bandwidth[0]: {torch.exp(model_x.kernel.log_gamma.data.reshape(-1)[0]):.4f}')
        
        print(f'Final: Loss = {loss.item():.4e}, Ridge = {model_x.ridge_lambda.item():.4e}, Bandwidth[0] = {torch.exp(model_x.kernel.log_gamma.data.reshape(-1)[0]):.4f}')


    def _fit_regressor(self, X, Y, model_x, model_y):
        """ Select the kernel parameters in X->Y regression with leave-one-out criterion. """

        # Compute and store the kernel matrices for Y
        model_y.set_kernel_matrix(Y)
        K_YY = model_y.kernel_matrix
        if model_x._kernel_matrix is not None:
            raise ValueError("Kernel matrix has already been computed.")
        # Select the kernel parameters for regressor with leave-one-out criterion
        if model_x.is_trainable:
            self._select_kernel_parameters(X, K_YY, model_x)
        # Compute and store the kernel matrices for X
        model_x.set_kernel_matrix(X)


    def get_conditional_mean(self, x, y, model_x, model_y):
        """
        Get the kernel matrix of y centered on the estimted conditional mean of y given x.
        K_yy_centered = <k(y,)-mu(k(y,)|x), k(y',)-mu(k(y',)|x')>
                      = K_yy + M + M.T
        M = (1/2 K_xX @ K_XX_inv_K_YY - K_yY) @ K_XX_inv @ K_xX.T
        """

        # Compute the kernel matrices for y
        K_yy = model_y(y, y)
        # Compute the inverse kernel matrices for X
        K_XX_inv, K_XX_inv_K_YY = solve_regularized_kernel_matrix_system(model_x.kernel_matrix, model_y.kernel_matrix, 
                                                            model_x.ridge_lambda)
        # Compute the matrix M
        K_xX, K_yY = model_x(x), model_y(y)
        M = (0.5 * K_xX @ K_XX_inv_K_YY - K_yY) @ K_XX_inv @ K_xX.T

        # Compute the centered kernel matrix for y
        K_yy_centered = K_yy + M + M.T

        return K_yy_centered.detach()


    def compute_statistic(self, a, b, c, return_matrices=False, return_var=False):
        """
        Compute the kernel conditional independence test statistic and variance.
        Return the test statistic.
        return_matrices (bool): If True, also return the centered kernel matrices.
        return_var (bool): If True, also return the variance of the test statistic.
        """
        # Compute the centered kernel matrices for a and b given c
        K_aa_centered = self.get_conditional_mean(c, a, self.kernel_ca, self.kernel_a)
        K_bb_centered = self.get_conditional_mean(c, b, self.kernel_cb, self.kernel_b)

        if self.kernel_c._kernel_matrix is not None:
            raise ValueError("Kernel matrix of C should not be set.")
        K_cc = self.kernel_c(c)

        K_bb_centered_cc = K_cc * K_bb_centered
        statistic_value = compute_hsic(K_aa_centered, K_bb_centered_cc, self.is_hsic_baised)

        if return_matrices:
            return statistic_value, K_aa_centered, K_bb_centered_cc
        if return_var:
            # Compute the biased estimation of the variance of kci statistic
            hh = K_aa_centered * K_bb_centered_cc
            n_var = 4 * ((hh.mean(0)**2).mean() - (hh.mean())**2)
            return statistic_value, n_var

        return statistic_value


    def maximize_test_power(self, a, b, c, var_offset=0., ):
        """
        Maximize the test power by learning the kernel parameters of the conditional independence test.
        """
        optimizer = optim.Adam(self.kernel_c.parameters(), lr=self.lr)

        self.kernel_c.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Compute statistic and variance
            kci_val, kci_var = self.compute_statistic(a, b, c, return_matrices=False, return_var=True) # statistic_value, n_variance
            kci_std = torch.sqrt(kci_var+torch.tensor([var_offset], device=self.device))
            # Compute loss
            loss = - (kci_val / kci_std)


            loss.backward()
            optimizer.step()

            if epoch%50 == 0 and self.verbose:
                print(f"Epoch [{epoch + 1}/{self.epochs}]: kci_value = {kci_val.item():.4e}, "
                        f"kci_std = {kci_std.item():.4e}, Loss = {loss.item():.4e}, "
                        f"Bandwidth[0] = {torch.exp(self.kernel_c.kernel.log_gamma.data.reshape(-1)[0]):.4f}")

        kci_val, kci_var = self.compute_statistic(a, b, c, return_matrices=False, return_var=True) # statistic_value, n_variance
        kci_std = torch.sqrt(kci_var+torch.tensor([var_offset], device=self.device))

        print(f"Final: kci_value = {kci_val.item():.4e}, "
                f"kci_std = {kci_std.item():.4e}, "
                f"Bandwidth[0] = {torch.exp(self.kernel_c.kernel.log_gamma.data.reshape(-1)[0]):.4f}")

        self.kernel_c.eval()


    def fit(self, a_train, b_train, c_train):
        """
        Learn the kernel parameters of the conditional independence test,
        and fit the regressions.
        """

        self._fit_regressor(c_train, a_train, self.kernel_ca, self.kernel_a)
        self._fit_regressor(c_train, b_train, self.kernel_cb, self.kernel_b)

        # Maximize the test power
        if self.kernel_c.is_trainable:
            self.maximize_test_power(a_train, b_train, c_train, var_offset=0.)


    def compute_p_value(self, a_test, b_test, c_test):
        """
        Compute the p-value of the conditional independence test.
        """
        # Compute the statistic
        kci_value, k_aa, k_bb = self.compute_statistic(a_test, b_test, c_test, return_matrices=True)

        # Compute the p-value
        p_value = compute_pval(kci_value.detach(), k_aa.detach(), k_bb.detach(), is_hsic_biased=self.is_hsic_baised, 
                               pval_approx_type=self.pval_approx_type, n_samples=self.n_wild_bootstrap_samples)

        return p_value
    