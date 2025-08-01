import numpy as np

class Sequential(object):
    '''
        The sequential model is a list of layers in it's order,
        the __init__() method takes a list of layers.

        A layer must have forward() and backward() method,
        the backward() method of each layer returns
            - grad_input
            - grad_weight (optional, or other names)
            - grad_bias (optional, or other names)

        For flexible implementations of the optimizers, the sum of param grads
        will be stored and accumulated after calling Sequential.backward().

        The last layer must be criterion (CrossEntropyLossWithSoftmax).
    '''
    def __init__(self, layers):
        self.layers = layers
        self.param_grads = None # will be a list of tuples containing tensors
    
    def forward(self, input, gt_label, train_mode=True):
        x = input
        # forward the layers except the last one
        for l in self.layers[:-1]:
            # print(x.shape)
            # print(type(l))
            if hasattr(l, 'train_mode'):  # 检查是否有train_mode参数
                x = l.forward(x, train_mode)
            else:
                x = l.forward(x)
        # the last one is criterion (CrossEntropyLossWithSoftmax)
        loss = self.layers[-1].forward(x, gt_label)
        return self.layers[-1].prob, loss

    def backward(self, grad):
        grads = []
        # 确保初始梯度是2D的
        if grad.ndim == 1:
            grad = grad.reshape(-1, 1)  # (N,) -> (N, 1)
            
        for l in reversed(self.layers):
            # 打印当前层和梯度形状，用于调试
            # print(f"Layer: {type(l).__name__}, Gradient shape: {grad.shape}")
            
            bwd_ret = l.backward(grad)
            if isinstance(bwd_ret, tuple):
                grad = bwd_ret[0]
                if len(bwd_ret) > 1:  # have trainable params
                    grad_params = list(bwd_ret[1])
                    grads.append(grad_params)
            else:  # only grad_input
                grad = bwd_ret
                grads.append([])
                
            # 检查梯度是否包含nan
            if np.isnan(grad).any():
                print(f"Warning: NaN gradient detected in {type(l).__name__}")
                
        grads.reverse()
        self.param_grads = grads