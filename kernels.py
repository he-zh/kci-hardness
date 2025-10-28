import torch
import torch.nn as nn
from utils import compute_pdist_sq

class Kernel(nn.Module):
    def __init__(self, kernel_type='linear', gamma=1.0, degree=3., coef0=1., is_trainable=False):
        super(Kernel, self).__init__()
        self.kernel_type = kernel_type
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma, dtype=torch.float32)), requires_grad=is_trainable)
        self.log_degree = nn.Parameter(torch.log(torch.tensor(degree, dtype=torch.float32)), requires_grad=is_trainable)
        self.coef0 = nn.Parameter(torch.tensor(coef0, dtype=torch.float32), requires_grad=is_trainable)
        

    def forward(self, X1, X2=None):
        if X2 is None:
            X2 = X1 # remove .clone()

        if self.kernel_type == 'linear':
            return torch.matmul(X1, X2.T)
        elif self.kernel_type == 'rbf':
            # exp(-1/(2*gamma) * ||x - y||^2)
            return torch.exp(-compute_pdist_sq(X1 / torch.exp(self.log_gamma/2), X2 / torch.exp(self.log_gamma/2)) / 2)
        elif self.kernel_type == 'polynomial':
            return (torch.exp(self.log_gamma) * torch.matmul(X1, X2.T) + self.coef0) ** torch.exp(self.log_degree)
        elif self.kernel_type == 'sigmoid':
            return torch.tanh(torch.exp(self.log_gamma) * torch.matmul(X1, X2.T) + self.coef0)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")



class BaseModel(nn.Module):
    """
    Base model class for kernel functions with feature extraction.

    Args:
        kernel_type (str): Type of kernel function to use.
        gamma (float): Parameter for the RBF, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel. Ignored by other kernels.
        is_trainable (bool): Whether to learn the kernel parameters
            from the data.
        degree : Degree of the polynomial kernel. Ignored by other kernels.
        coef0 : Zero coefficient for polynomial and sigmoid kernels. Ignored by other kernels.
        feature_extractor_parameters (dict): Parameters for the feature extractor.

    """
    def __init__(self, kernel_type='linear', gamma=1.0, gamma_dim=1, degree=3., coef0=1., ridge_lambda=1.0,
                 feature_extractor=None, is_trainable=False, **kwargs):
        super(BaseModel, self).__init__()
        gamma = gamma if gamma_dim == 1 else [[gamma]*gamma_dim]
        self.ridge_lambda = nn.Parameter(torch.tensor(ridge_lambda, dtype=torch.float32), requires_grad=is_trainable)
        self.log_ridge_lambda = nn.Parameter(torch.log(torch.tensor(ridge_lambda, dtype=torch.float32)), requires_grad=is_trainable)
        self.kernel = Kernel(kernel_type=kernel_type, gamma=gamma, degree=degree, coef0=coef0, 
                             is_trainable=is_trainable)
        self.feature_extractor = feature_extractor if feature_extractor is not None \
                                    else nn.Identity()
        self.is_trainable = is_trainable
        self._train_feature = None
        self._kernel_matrix = None

        print(f"Initialized model with kernel_type={kernel_type}, is_trainable={is_trainable}")

    @property
    def train_feature(self):
        if self._train_feature is None:
            raise ValueError("Training features have not been set. Call `set_kernel_matrix` first.")
        return self._train_feature

    @property
    def kernel_matrix(self):
        if self._kernel_matrix is None:
            raise ValueError("Kernel matrix has not been computed. Call `set_kernel_matrix` first.")
        return self._kernel_matrix

    def set_kernel_matrix(self, train_X):
        """
        Sets the training features and kernel matrix.
        For trainable models, this function should be called after training.
        """
        # if train_X.shape[0] > 2000:
        #     indices = torch.randperm(train_X.shape[0])[:2000]
        #     train_X = train_X[indices]

        self._train_feature = self.feature_extractor(train_X).detach()
        self._kernel_matrix = self.kernel(self._train_feature).detach()


    def forward(self, X1, X2=None):
        """
        Compute the kernel matrix for the input data
        """
        features_X1 = self.feature_extractor(X1)
        if X2 is not None:
            # K(X1, X2)
            features_X2 = self.feature_extractor(X2)
        else:
            if self._train_feature is not None:
                # K(X1, train_X)
                features_X2 = self.train_feature
            else:
                # K(X1, X1)
                features_X2 = features_X1

        return self.kernel(features_X1, features_X2)

class LinearModel(BaseModel):
    def __init__(self, input_dim, ridge_lambda=1.0, is_trainable=False, **kwargs):
        super(LinearModel, self).__init__(kernel_type='linear', gamma=1.0, gamma_dim=input_dim, 
                                      ridge_lambda=ridge_lambda, 
                                      feature_extractor=None, 
                                      is_trainable=is_trainable)
        
class RBFModel(BaseModel):
    def __init__(self, input_dim, gamma=1.0, ridge_lambda=1.0, is_trainable=False, **kwargs):
        super(RBFModel, self).__init__(kernel_type='rbf', gamma=gamma, gamma_dim=input_dim, 
                                      ridge_lambda=ridge_lambda, 
                                      feature_extractor=None, 
                                      is_trainable=is_trainable)

