import torch
import torch.nn.functional as F
from collections import deque
import networkx as nx

class MyTensor:
    def __init__(self, data, requires_grad=False, _op=None):
        if not isinstance(data, torch.Tensor):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            self.data = data.clone().detach()

        self.requires_grad = requires_grad
        self.grad = None
        self._op = _op
        self._is_leaf = requires_grad and _op is None
        # Add a unique ID for using as a node in the graph
        self.id = id(self)

    def zero_grad(self):
        """Sets gradients to None."""
        self.grad = None

    # --- Standard PyTorch-like Tensor Operations ---
    def __add__(self, other):
        return Add.apply(self, other)

    def __mul__(self, other):
        return Multiply.apply(self, other)

    def __sub__(self, other):
        return Subtract.apply(self, other)

    def __truediv__(self, other):
        return Divide.apply(self, other)

    def matmul(self, other):
        return MatMul.apply(self, other)

    def sum(self, dim=None, keepdim=False):
        return Sum.apply(self, dim, keepdim)

    def mean(self, dim=None, keepdim=False):
        return Mean.apply(self, dim, keepdim)

    def relu(self):
        return ReLU.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def tanh(self):
        return Tanh.apply(self)

    # --- The Core Backpropagation Algorithm: Efficient Graph-based Approach ---
    def backward(self, grad_output=None):
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require grad. Set requires_grad=True to enable.")

        # 1. Build the computation graph using networkx
        # Nodes are the unique IDs of MyTensor instances
        graph = nx.DiGraph()
        q = deque([self])
        visited = {self}
        while q:
            tensor_node = q.popleft()
            if tensor_node._op:
                graph.add_node(tensor_node.id, op=tensor_node._op)
                for input_tensor in tensor_node._op.inputs:
                    if isinstance(input_tensor, MyTensor):
                        # Edge from input tensor to the current tensor
                        graph.add_edge(input_tensor.id, tensor_node.id)
                        if input_tensor not in visited:
                            q.append(input_tensor)
                            visited.add(input_tensor)
        
        # Add leaf nodes that may not have ops
        for node in visited:
            if node.id not in graph:
                graph.add_node(node.id, op=node._op)


        # 2. Get the operations in reverse topological order
        topo_order_ids = list(nx.topological_sort(graph))
        
        # Create a mapping from ID back to the tensor object
        id_to_tensor = {t.id: t for t in visited}

        # 3. Initialize gradients with a standard dictionary
        gradients = {}
        if grad_output is None:
            # Ensure grad_output matches the shape of the loss tensor's data
            grad_output = torch.ones_like(self.data)
        gradients[self.id] = grad_output

        # 4. Iterate and propagate gradients
        for tensor_id in reversed(topo_order_ids):
            tensor_node = id_to_tensor.get(tensor_id)
            if not tensor_node or not tensor_node._op:
                continue

            grad_for_op_output = gradients.get(tensor_id)
            if grad_for_op_output is None:
                continue

            # Call the backward method of the operation
            grads_from_op = tensor_node._op.backward(grad_for_op_output)

            # Distribute gradients to inputs
            for i, input_tensor in enumerate(tensor_node._op.inputs):
                if isinstance(input_tensor, MyTensor) and input_tensor.requires_grad:
                    grad_for_input = grads_from_op[i]
                    if input_tensor._is_leaf:
                        if input_tensor.grad is None:
                            input_tensor.grad = grad_for_input.clone()
                        else:
                            input_tensor.grad += grad_for_input
                    else:
                        # Safely accumulate gradients in the dictionary
                        if input_tensor.id not in gradients:
                            gradients[input_tensor.id] = grad_for_input.clone()
                        else:
                            gradients[input_tensor.id] += grad_for_input


class _Context:
    """Simple context manager to save tensors/values for the backward pass."""
    def __init__(self):
        self._saved_tensors = ()

    def save_for_backward(self, *tensors):
        self._saved_tensors = tuple(t.detach() if isinstance(t, torch.Tensor) else t for t in tensors)

    @property
    def saved_tensors(self):
        return self._saved_tensors


class Operation:
    def __init__(self, *inputs):
        self.inputs = inputs
        self.output_tensor = None
        self.ctx = _Context()

    @classmethod
    def apply(cls, *args):
        if cls.__name__ in ['Sum', 'Mean']:
            my_tensor = args[0]
            extra_args = args[1:] if len(args) > 1 else ()
            if not isinstance(my_tensor, MyTensor):
                my_tensor = MyTensor(my_tensor)
            requires_grad = my_tensor.requires_grad
            op_instance = cls(my_tensor, *extra_args)
            output_data = op_instance.forward(my_tensor.data)
        else:
            processed_inputs = []
            requires_grad = False
            for inp in args:
                if not isinstance(inp, MyTensor):
                    inp = MyTensor(inp)
                if inp.requires_grad:
                    requires_grad = True
                processed_inputs.append(inp)
            op_instance = cls(*processed_inputs)
            output_data = op_instance.forward(*[i.data for i in processed_inputs])

        output_tensor = MyTensor(output_data, requires_grad=requires_grad, _op=op_instance)
        op_instance.output_tensor = output_tensor
        return output_tensor

    def forward(self, *input_data):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

# --- Common Operation Subclasses (Unchanged) ---

class Add(Operation):
    def forward(self, a_data, b_data):
        return a_data + b_data
    def backward(self, grad_output):
        return grad_output, grad_output

class Multiply(Operation):
    def forward(self, a_data, b_data):
        self.ctx.save_for_backward(a_data, b_data)
        return a_data * b_data
    def backward(self, grad_output):
        a_data, b_data = self.ctx.saved_tensors
        return grad_output * b_data, grad_output * a_data

class Subtract(Operation):
    def forward(self, a_data, b_data):
        return a_data - b_data
    def backward(self, grad_output):
        return grad_output, -grad_output

class Divide(Operation):
    def forward(self, a_data, b_data):
        self.ctx.save_for_backward(a_data, b_data)
        return a_data / b_data
    def backward(self, grad_output):
        a_data, b_data = self.ctx.saved_tensors
        return grad_output / b_data, -grad_output * a_data / (b_data * b_data)

class MatMul(Operation):
    def forward(self, a_data, b_data):
        self.ctx.save_for_backward(a_data, b_data)
        return torch.matmul(a_data, b_data)
    def backward(self, grad_output):
        a_data, b_data = self.ctx.saved_tensors
        grad_a = torch.matmul(grad_output, b_data.T)
        grad_b = torch.matmul(a_data.T, grad_output)
        return grad_a, grad_b

class Sum(Operation):
    def __init__(self, input_tensor, dim=None, keepdim=False):
        super().__init__(input_tensor)
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, input_data):
        self.ctx.save_for_backward(input_data)
        return input_data.sum(dim=self.dim, keepdim=self.keepdim)
    def backward(self, grad_output):
        input_data, = self.ctx.saved_tensors
        input_shape = input_data.shape
        if self.dim is not None and not self.keepdim:
            if isinstance(self.dim, int):
                dims_to_unsqueeze = [self.dim]
            else:
                dims_to_unsqueeze = list(self.dim)
            for d in sorted(dims_to_unsqueeze):
                grad_output = grad_output.unsqueeze(d)
        grad_input = grad_output.expand(input_shape)
        return (grad_input,)

class Mean(Operation):
    def __init__(self, input_tensor, dim=None, keepdim=False):
        super().__init__(input_tensor)
        self.dim = dim
        self.keepdim = keepdim
    def forward(self, input_data):
        if self.dim is not None:
            if isinstance(self.dim, int):
                num_elements_reduced = input_data.shape[self.dim]
            else:
                num_elements_reduced = 1
                for d in self.dim:
                    num_elements_reduced *= input_data.shape[d]
        else:
            num_elements_reduced = input_data.numel()
        self.ctx.save_for_backward(input_data, torch.tensor(num_elements_reduced, dtype=torch.float32))
        return input_data.mean(dim=self.dim, keepdim=self.keepdim)
    def backward(self, grad_output):
        input_data, num_elements_reduced_tensor = self.ctx.saved_tensors
        input_shape = input_data.shape
        num_elements_reduced = num_elements_reduced_tensor.item()
        if self.dim is not None and not self.keepdim:
            if isinstance(self.dim, int):
                dims_to_unsqueeze = [self.dim]
            else:
                dims_to_unsqueeze = list(self.dim)
            for d in sorted(dims_to_unsqueeze):
                grad_output = grad_output.unsqueeze(d)
        grad_input = (grad_output / num_elements_reduced).expand(input_shape)
        return (grad_input,)

class ReLU(Operation):
    def forward(self, x_data):
        self.ctx.save_for_backward(x_data)
        return torch.relu(x_data)
    def backward(self, grad_output):
        x_data, = self.ctx.saved_tensors
        return (grad_output * (x_data > 0).to(x_data.dtype),)

class Sigmoid(Operation):
    def forward(self, x_data):
        output = torch.sigmoid(x_data)
        self.ctx.save_for_backward(output)
        return output
    def backward(self, grad_output):
        output, = self.ctx.saved_tensors
        return (grad_output * (output * (1 - output)),)

class Tanh(Operation):
    def forward(self, x_data):
        output = torch.tanh(x_data)
        self.ctx.save_for_backward(output)
        return output
    def backward(self, grad_output):
        output, = self.ctx.saved_tensors
        return (grad_output * (1 - output.pow(2)),)


# --- Test the fixed implementation ---
if __name__ == "__main__":
    # Create leaf tensors
    a = MyTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), requires_grad=True)
    b = MyTensor(torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32), requires_grad=True)
    bias = MyTensor(torch.tensor([0.1, 0.2], dtype=torch.float32), requires_grad=True)

    print(f"a:\n{a.data}\n")
    print(f"b:\n{b.data}\n")
    print(f"bias:\n{bias.data}\n")

    # --- Building a more complex DAG ---
    linear1_out = a.matmul(b)
    activated1 = linear1_out.relu()
    c = activated1 + a
    d = c.sum(dim=1)
    loss = d.mean()
    print(f"Loss (d.mean()):\n{loss.data}\n")

    # Zero gradients before backward pass
    a.zero_grad()
    b.zero_grad()
    bias.zero_grad()

    # Perform backward pass
    print("Starting backward pass...")
    loss.backward()
    print("Backward pass completed!")

    # Check gradients
    print(f"\nGradient of Loss with respect to a:\n{a.grad}\n")
    print(f"Gradient of Loss with respect to b:\n{b.grad}\n")
    print(f"Gradient of Loss with respect to bias:\n{bias.grad}\n")

    # --- Verification with PyTorch's autograd ---
    print("\n--- Verifying with PyTorch's autograd ---")
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32, requires_grad=True)

    linear1_out_torch = a_torch.matmul(b_torch)
    activated1_torch = F.relu(linear1_out_torch)
    c_torch = activated1_torch + a_torch
    d_torch = c_torch.sum(dim=1)
    loss_torch = d_torch.mean()
    loss_torch.backward()

    print(f"PyTorch's a.grad:\n{a_torch.grad}\n")
    print(f"PyTorch's b.grad:\n{b_torch.grad}\n")
    
    # --- Final Check ---
    print("A gradients match PyTorch:", torch.allclose(a.grad, a_torch.grad, atol=1e-6) if a.grad is not None else "False (grad is None)")
    print("B gradients match PyTorch:", torch.allclose(b.grad, b_torch.grad, atol=1e-6) if b.grad is not None else "False (grad is None)")