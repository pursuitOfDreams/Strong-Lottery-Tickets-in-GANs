import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

def get_subnet(scores, k):
    out = scores.clone()
    _, idx = scores.flatten().sort()
    j = int((1 - k) * scores.numel())

    # flat_out and out access the same memory.
    flat_out = out.flatten()
    flat_out[idx[:j]] = False
    flat_out[idx[j:]] = True

    return out
   
# logic for model pruning using BasePruningMethod
class EdgePop(prune.BasePruningMethod):
  PRUNING_TYPE = 'unstructured'
  def __init__(self):
    super().__init__()
    self.k = 0.1


  def compute_mask(self, t, default_mask):
    mask = t.clone()
    _, idx = t.flatten().sort()
    j = int((1 - self.k) * t.numel())

    # flat_out and out access the same memory.
    flat_out = mask.flatten()
    flat_out[idx[:j]] = False
    flat_out[idx[j:]] = True

    return mask

  @classmethod
  def apply(cls, module, name, *args, importance_scores=None, **kwargs):

    def _get_composite_method(cls, module, name, *args, **kwargs):
        
        found = 0
        # there should technically be only 1 hook with hook.name == name
        # assert this using `found`
        hooks_to_remove = []
        for k, hook in module._forward_pre_hooks.items():
            hooks_to_remove.append(k)
            found += 1
        
        for k in hooks_to_remove:
            del module._forward_pre_hooks[k]

        # Apply the new pruning method, either from scratch or on top of
        # the previous one.
        method = cls(*args, **kwargs)  # new pruning
        # Have the pruning method remember what tensor it's been applied to
        method._tensor_name = name

        return method, found
    method, found = _get_composite_method(cls, module, name, *args, **kwargs)
    orig = getattr(module, name)
    if importance_scores is not None:
        assert (
            importance_scores.shape == orig.shape
        ), "importance_scores should have the same shape as parameter \
            {} of {}".format(
            name, module
        )
    else:
        importance_scores = orig

    if found <= 1:
        # copy `module[name]` to `module[name + '_orig']`
        module.register_parameter(name + "_orig", orig)
        # temporarily delete `module[name]`
        del module._parameters[name]
        default_mask = torch.ones_like(orig)  
    else:
        default_mask = (
            getattr(module, name + "_mask")
            .detach()
            .clone(memory_format=torch.contiguous_format)
        )

    try:
        # get the final mask, computed according to the specific method
       
        mask = method.compute_mask(importance_scores, default_mask=default_mask)
        # reparametrize by saving mask to `module[name + '_mask']`...
        module.register_buffer(name + "_mask", mask)
        # ... and the new pruned tensor to `module[name]`
        setattr(module, name, method.apply_mask(module))
        # associate the pruning method to the module via a hook to
        # compute the function before every forward() (compile by run)
        module.register_forward_pre_hook(method)


    except Exception as e:
        if found <=1 :
            orig = getattr(module, name + "_orig")
            module.register_parameter(name, orig)
            del module._parameters[name + "_orig"]
        raise e

    return method
  def remove_pruning(self, module, name):

    
    if hasattr(module, name):
        delattr(module, name)
    orig = module._parameters[name + "_orig"]
    del module._parameters[name + "_orig"]
    del module._buffers[name + "_mask"]
    setattr(module, name, orig)


def prune_model(model, algo="ep"):
  ep = EdgePop()
  for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
      try:
        if algo =="ep":
          ep.apply(module, name = 'weight', importance_scores = module.scores.abs())
          ep.apply(module, name = 'bias', importance_scores = module.bias_scores.abs())
        else:
          ep.remove_pruning(module)
          prune.random_unstructured(module, name ='weight', amount=0.1 )
          prune.random_unstructured(module, name ='bias', amount=0.1 )
      except Exception as e:
        print(e)
    elif isinstance(module, nn.Linear):
      try:
        if algo =="ep":
          ep.apply(module, name = 'weight', importance_scores = module.scores.abs())
          ep.apply(module, name = 'bias', importance_scores = module.bias_scores.abs())
        else:
          prune.random_unstructured(module, name ='weight', amount=0.1 )
          prune.random_unstructured(module, name ='bias', amount=0.1 )
      except Exception as e:
        print(1)

def remove_pruning(model, algo="ep"):
    ep = EdgePop()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                if algo =="ep":
                    ep.remove(module)
                    ep.apply(module)
                else:
                    ep.remove_pruning(module)
                    prune.random_unstructured(module, name ='weight', amount=0.1 )
                    prune.random_unstructured(module, name ='bias', amount=0.1 )
            except Exception as e:
                print(e)
        elif isinstance(module, nn.Linear):
            try:
                if algo =="ep":
                    ep.remove(module)
                    ep.remove(module)
                else:
                    prune.random_unstructured(module, name ='weight', amount=0.1 )
                    prune.random_unstructured(module, name ='bias', amount=0.1 )
            except Exception as e:
                print(1)