import numpy as np
from math import sqrt
import pylayer as L  # 导入基础层
from sequential import Sequential  # 导入Sequential类

class PatchEmbedding(object):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        self.img_size = img_size
        self.patch_size = patch_size
        # 计算实际的patch数量：对于28x28的图像和7x7的patch，应该是4x4=16
        self.n_patches = (img_size // patch_size) ** 2  # 16 = 4x4 patches
        print(f"Creating PatchEmbedding with {self.n_patches} patches ({img_size//patch_size}x{img_size//patch_size})")
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        # 计算每个patch的特征维度
        self.patch_dim = in_channels * patch_size * patch_size  # 1 * 7 * 7 = 49
        
        # 使用Linear层而不是Conv2d来实现patch embedding
        self.proj = L.Linear(self.patch_dim, embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape  # (B, C, H, W) = (64, 1, 28, 28)
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}*{W}) doesn't match expected size ({self.img_size}*{self.img_size})"
        assert C == self.in_channels, f"Input channels ({C}) doesn't match expected channels ({self.in_channels})"
        
        # 将图像分成patches
        # 1. 重塑以提取patches: (B, C, H, W) -> (B, C, n_h, patch_size, n_w, patch_size)
        x = x.reshape(B, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
        # 2. 调整维度顺序: -> (B, n_h, n_w, patch_size, patch_size, C)
        x = x.transpose(0, 2, 4, 3, 5, 1)
        # 3. 合并patches维度并展平每个patch: -> (B, n_patches, patch_dim)
        x = x.reshape(B, self.n_patches, self.patch_dim)
        
        # 保存输入形状用于反向传播
        self.input_shape = x.shape  # 应该是 (B, 16, patch_dim)
        
        # 线性投影到embed_dim
        x_reshaped = x.reshape(-1, self.patch_dim)  # (B*n_patches, patch_dim)
        x = self.proj.forward(x_reshaped)  # (B*n_patches, embed_dim)
        x = x.reshape(B, self.n_patches, self.embed_dim)  # (B, n_patches, embed_dim)
        
        return x
        
    def backward(self, grad_output):
        B, N, C = grad_output.shape
        assert C == self.embed_dim, f"Gradient channels ({C}) doesn't match embed_dim ({self.embed_dim})"
        assert N == self.n_patches, f"Gradient patches ({N}) doesn't match n_patches ({self.n_patches}), got shape {grad_output.shape}"
        
        # 线性层的反向传播
        grad_reshaped = grad_output.reshape(-1, self.embed_dim)  # (B*n_patches, embed_dim)
        grad_output, grad_w, grad_b = self.proj.backward(grad_reshaped)  # (B*n_patches, patch_dim)
        grad_output = grad_output.reshape(B, self.n_patches, self.patch_dim)  # (B, n_patches, patch_dim)
        
        # 重建原始图像形状
        H = W = self.img_size
        n_h = n_w = H // self.patch_size
        grad_output = grad_output.reshape(B, n_h, n_w, self.patch_size, self.patch_size, self.in_channels)
        grad_output = grad_output.transpose(0, 5, 1, 3, 2, 4)  # (B, C, n_h, patch_size, n_w, patch_size)
        grad_output = grad_output.reshape(B, self.in_channels, H, W)  # (B, C, H, W)
        
        return grad_output, grad_w, grad_b

class MultiHeadAttention(object):
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Q, K, V 投影矩阵，使用Xavier初始化
        scale = np.sqrt(2.0 / (embed_dim + embed_dim))  # Xavier初始化
        self.q_proj = L.Linear(embed_dim, embed_dim)
        self.k_proj = L.Linear(embed_dim, embed_dim)
        self.v_proj = L.Linear(embed_dim, embed_dim)
        self.out_proj = L.Linear(embed_dim, embed_dim)
        
        # 缩放初始权重
        self.q_proj.weight *= scale
        self.k_proj.weight *= scale
        self.v_proj.weight *= scale
        self.out_proj.weight *= scale
        
        # 初始化为接近单位矩阵
        eye = np.eye(embed_dim)
        self.q_proj.weight += eye * 0.1
        self.k_proj.weight += eye * 0.1
        self.v_proj.weight += eye * 0.1
        self.out_proj.weight += eye * 0.1
        
    def forward(self, x):
        # 输入: (B, N, C) = (64, 16, 64)
        B, N, C = x.shape
        
        # 线性投影: 先reshape成2D进行线性变换
        x_2d = x.reshape(-1, C)  # (B*N, C) = (1024, 64)
        q = self.q_proj.forward(x_2d)  # (B*N, C) = (1024, 64)
        k = self.k_proj.forward(x_2d)  # (B*N, C) = (1024, 64)
        v = self.v_proj.forward(x_2d)  # (B*N, C) = (1024, 64)
        
        # 重塑为多头形式
        q = q.reshape(B, N, self.num_heads, self.head_dim)  # (64, 16, 8, 8)
        k = k.reshape(B, N, self.num_heads, self.head_dim)  # (64, 16, 8, 8)
        v = v.reshape(B, N, self.num_heads, self.head_dim)  # (64, 16, 8, 8)
        
        q = q.transpose(0, 2, 1, 3)  # (B, num_heads, N, head_dim) = (64, 8, 16, 8)
        k = k.transpose(0, 2, 1, 3)  # (64, 8, 16, 8)
        v = v.transpose(0, 2, 1, 3)  # (64, 8, 16, 8)
        
        # 计算注意力，使用更大的缩放因子
        scale = np.sqrt(self.head_dim) * 2
        attn = np.matmul(q, k.transpose(0, 1, 3, 2)) / scale  # (64, 8, 16, 16)
        
        # 使用更稳定的softmax实现
        attn_max = np.max(attn, axis=-1, keepdims=True)
        exp_attn = np.exp(attn - attn_max)
        attn = exp_attn / (np.sum(exp_attn, axis=-1, keepdims=True) + 1e-6)
        
        # 应用注意力
        out = np.matmul(attn, v)  # (64, 8, 16, 8)
        
        # 重塑回原始形状
        out = out.transpose(0, 2, 1, 3)  # (64, 16, 8, 8)
        out = out.reshape(B, N, C)  # (64, 16, 64)
        
        # 输出投影
        out = out.reshape(B * N, C)  # (1024, 64)
        out = self.out_proj.forward(out)  # (1024, 64)
        out = out.reshape(B, N, C)  # (64, 16, 64)
        
        # 值范围检查
        if np.abs(out).max() > 10:
            out = out * (10 / np.abs(out).max())
        
        # 保存中间结果用于反向传播
        self.attn = attn
        self.q, self.k, self.v = q, k, v
        self.x_shape = (B, N, C)
        self.x_2d = x_2d
        
        return out
        
    def backward(self, grad_output):
        # 输入梯度: (B, N, C) = (64, 16, 64)
        B, N, C = self.x_shape
        
        # 输出投影的反向传播
        grad_output_2d = grad_output.reshape(-1, C)  # (1024, 64)
        grad_output_2d, grad_out_w, grad_out_b = self.out_proj.backward(grad_output_2d)
        grad_output = grad_output_2d.reshape(B, N, C)  # (64, 16, 64)
        
        # 重塑梯度为多头形式
        grad_output = grad_output.reshape(B, N, self.num_heads, self.head_dim)  # (64, 16, 8, 8)
        grad_output = grad_output.transpose(0, 2, 1, 3)  # (64, 8, 16, 8)
        
        # 注意力的反向传播
        grad_v = np.matmul(self.attn.transpose(0, 1, 3, 2), grad_output)  # (64, 8, 16, 8)
        grad_attn = np.matmul(grad_output, self.v.transpose(0, 1, 3, 2))  # (64, 8, 16, 16)
        
        grad_q = np.matmul(grad_attn, self.k) / sqrt(self.head_dim)  # (64, 8, 16, 8)
        grad_k = np.matmul(grad_attn.transpose(0, 1, 3, 2), self.q) / sqrt(self.head_dim)  # (64, 8, 16, 8)
        
        # 重塑回原始形状
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(B * N, -1)  # (1024, 64)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(B * N, -1)  # (1024, 64)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(B * N, -1)  # (1024, 64)
        
        # Q, K, V投影的反向传播
        grad_x_q, grad_q_w, grad_q_b = self.q_proj.backward(grad_q)
        grad_x_k, grad_k_w, grad_k_b = self.k_proj.backward(grad_k)
        grad_x_v, grad_v_w, grad_v_b = self.v_proj.backward(grad_v)
        
        # 合并梯度
        grad_x = (grad_x_q + grad_x_k + grad_x_v).reshape(B, N, -1)  # (64, 16, 64)
        
        # 返回元组
        grads = (
            grad_q_w, grad_q_b,    # Q投影
            grad_k_w, grad_k_b,    # K投影
            grad_v_w, grad_v_b,    # V投影
            grad_out_w, grad_out_b  # 输出投影
        )
        
        return grad_x, grads

class MLP(object):
    def __init__(self, in_features, hidden_features, out_features):
        # 使用更小的初始化范围
        self.fc1 = L.Linear(in_features, hidden_features)
        self.act = L.ReLU()
        self.fc2 = L.Linear(hidden_features, out_features)
        
        # 缩放初始权重
        self.fc1.weight *= 0.1
        self.fc2.weight *= 0.1
        
    def forward(self, x):
        # 添加值范围检查
        if np.abs(x).max() > 10:
            x = x * (10 / np.abs(x).max())
            
        x = self.fc1.forward(x)
        x = self.act.forward(x)
        
        # ReLU后也检查范围
        if np.abs(x).max() > 10:
            x = x * (10 / np.abs(x).max())
            
        x = self.fc2.forward(x)
        
        # 输出前最后检查一次范围
        if np.abs(x).max() > 10:
            x = x * (10 / np.abs(x).max())
            
        return x
        
    def backward(self, grad_output):
        # 梯度裁剪
        if np.abs(grad_output).max() > 10:
            grad_output = grad_output * (10 / np.abs(grad_output).max())
            
        # 反向传播
        grad_output, grad_fc2_w, grad_fc2_b = self.fc2.backward(grad_output)
        grad_output = self.act.backward(grad_output)
        grad_output, grad_fc1_w, grad_fc1_b = self.fc1.backward(grad_output)
        
        # 裁剪梯度
        grad_fc1_w = np.clip(grad_fc1_w, -1, 1)
        grad_fc1_b = np.clip(grad_fc1_b, -1, 1)
        grad_fc2_w = np.clip(grad_fc2_w, -1, 1)
        grad_fc2_b = np.clip(grad_fc2_b, -1, 1)
        
        return grad_output, (grad_fc1_w, grad_fc1_b, grad_fc2_w, grad_fc2_b)

class TransformerBlock(object):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.):
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = L.BatchNorm1d(embed_dim)
        self.norm2 = L.BatchNorm1d(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)
        
    def forward(self, x, train_mode=True):
        B, N, C = x.shape
        assert N == 16, f"Expected 16 patches (4x4), got {N}"  # 验证patch数量
        
        # 第一个残差连接
        identity = x
        x = self.norm1.forward(x, train_mode)  # (B, N, C)
        
        # 注意力层
        attn_out = self.attn.forward(x)  # (B, N, C)
        # 残差连接时进行缩放
        x = attn_out * 0.5 + identity * 0.5  # 使用加权和而不是直接相加
        
        # 第二个残差连接
        identity = x
        x = self.norm2.forward(x, train_mode)  # (B, N, C)
        
        # MLP层
        x_2d = x.reshape(-1, C)  # (B*N, C)
        mlp_out = self.mlp.forward(x_2d)  # (B*N, C)
        mlp_out = mlp_out.reshape(B, N, C)  # 恢复3D形状
        # 残差连接时进行缩放
        x = mlp_out * 0.5 + identity * 0.5
        
        # 值范围检查
        if np.abs(x).max() > 10:
            x = x * (10 / np.abs(x).max())
        
        return x
        
    def backward(self, grad_output):
        B, N, C = grad_output.shape
        assert N == 16, f"Expected 16 patches in gradient, got {N}"  # 验证patch数量
        
        # 第二个残差连接的反向传播
        grad_mlp = grad_output
        grad_identity2 = grad_output
        
        # MLP的反向传播
        grad_mlp = grad_mlp.reshape(-1, C)  # (B*N, C)
        grad_mlp, mlp_grads = self.mlp.backward(grad_mlp)
        grad_mlp = grad_mlp.reshape(B, N, C)  # 恢复3D形状
        
        # Norm2的反向传播
        grad_norm2, grad_norm2_g, grad_norm2_b = self.norm2.backward(grad_mlp)
        grad_output = grad_norm2 + grad_identity2
        
        # 第一个残差连接的反向传播
        grad_attn = grad_output
        grad_identity1 = grad_output
        
        # 注意力的反向传播
        grad_attn, attn_grads = self.attn.backward(grad_attn)
        grad_norm1, grad_norm1_g, grad_norm1_b = self.norm1.backward(grad_attn)
        grad_output = grad_norm1 + grad_identity1
        
        # 验证输出梯度形状
        assert grad_output.shape == (B, N, C), f"Invalid gradient shape: {grad_output.shape}, expected: ({B}, {N}, {C})"
        
        # 按照params_ref的结构组织梯度（展平的列表）
        grads = (
            *attn_grads,           # 8个梯度：Q,K,V,Out的weight和bias
            grad_norm1_g, grad_norm1_b,  # 2个梯度：norm1的gamma和beta
            grad_norm2_g, grad_norm2_b,  # 2个梯度：norm2的gamma和beta
            *mlp_grads            # 4个梯度：两个Linear层的weight和bias
        )
        
        return grad_output, grads

class GlobalAveragePooling(object):
    def __init__(self):
        pass
        
    def forward(self, x):
        # x: (B, N, C)
        self.input_shape = x.shape
        return np.mean(x, axis=1)  # (B, C)
        
    def backward(self, grad_output):
        # grad_output: (B, C) or (B, C, 1)
        B, N, C = self.input_shape
        
        # 确保grad_output是2D的
        if grad_output.ndim == 3:
            grad_output = grad_output.squeeze(-1)
            
        # 检查形状
        if grad_output.shape != (B, C):
            print(f"Warning: Unexpected gradient shape in GlobalAveragePooling: {grad_output.shape}, expected: ({B}, {C})")
            grad_output = grad_output.reshape(B, C)
            
        # 将梯度平均分配给每个位置
        grad_input = np.repeat(grad_output[:, np.newaxis, :], N, axis=1) / N  # (B, N, C)
        
        # 验证输出形状
        assert grad_input.shape == (B, N, C), f"Invalid gradient shape: {grad_input.shape}, expected: ({B}, {N}, {C})"
        
        return grad_input

class ViT(Sequential):  # 改回继承Sequential
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64,
                 num_heads=8, depth=6, mlp_ratio=4., num_classes=10):
        # 计算patch的数量
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # 构建网络层
        layers = []
        
        # Patch Embedding
        layers.append(PatchEmbedding(img_size, patch_size, in_channels, embed_dim))
        
        # 添加位置编码
        pos_embed = np.zeros((1, self.num_patches, embed_dim))
        for i in range(self.num_patches):
            for j in range(embed_dim):
                pos_embed[0, i, j] = np.sin(i / 10000 ** (2 * j / embed_dim)) if j % 2 == 0 else np.cos(i / 10000 ** (2 * (j-1) / embed_dim))
        self.pos_embed = pos_embed
        
        # Transformer Encoder
        for _ in range(depth):
            layers.append(TransformerBlock(embed_dim, num_heads, mlp_ratio))
        
        # 分类头：使用全局平均池化
        layers.append(GlobalAveragePooling())  # (B, N, C) -> (B, C)
        
        # 最后的线性层使用特殊初始化
        final_linear = L.Linear(embed_dim, num_classes)
        final_linear.weight = np.random.randn(embed_dim, num_classes).astype(np.float32)
        final_linear.bias = np.zeros(num_classes).astype(np.float32)

    
        layers.append(final_linear)
        
        # 损失层
        layers.append(L.CrossEntropyLossWithSoftmax())
        
        super().__init__(layers)  # 调用父类构造函数
        
    def forward(self, input, gt_label, train_mode=True):
        x = input
        # forward the layers except the last one
        for i, l in enumerate(self.layers[:-2]):  # 除了最后两层
            # 在第一层之后添加位置编码
            if i == 0:
                x = l.forward(x)
                x = x + self.pos_embed
            else:
                if hasattr(l, 'train_mode'):  # 检查是否有train_mode参数
                    x = l.forward(x, train_mode)
                else:
                    x = l.forward(x)
        
        # 最后的线性层
        x = self.layers[-2].forward(x)
        print(f"\nFinal linear layer output range: min={x.min():.4f}, max={x.max():.4f}")
        print(f"Example logits:\n{x[0]}")
        
        # 损失层
        loss = self.layers[-1].forward(x, gt_label)
        return self.layers[-1].prob, loss 