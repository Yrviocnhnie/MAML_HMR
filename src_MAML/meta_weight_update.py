import torch
import copy

from collections import OrderedDict
from torchmeta.modules import MetaModule


def gradient_update_parameters(model, loss, updated_params = None, params=None, 
                               step_size=0.5):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.
    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.
    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.
    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))


    if params is None:
        params = model.parameters()
        updated_params = OrderedDict(model.meta_named_parameters())
        # updated_params = OrderedDict()
    # params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss, params)

    params_new = OrderedDict()
    updated_params_grad = []

    # for (name, param), grad in zip(params.items(), grads):
    #     updated_params[name] = param - step_size * grad
        
    
    # grad = torch.autograd.grad(loss, params)
    # tuples = zip(grads, params)
        
    for (name, param), grad in zip(updated_params.items(), grads):
        params_new[name] = param - step_size * grad
        updated_params_grad.append(updated_params[name])
    
        
    # updated_params_grad = list(map(lambda p: p[1] - step_size*p[0], tuples))
            

    return updated_params, updated_params_grad