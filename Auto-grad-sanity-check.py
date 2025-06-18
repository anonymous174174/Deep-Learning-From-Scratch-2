import torch
import torch.nn.functional as F

class MyTensor:
    def __init__(self, data, requires_grad=False, _op=None):
        if not isinstance(data, torch.Tensor):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            # Detach from PyTorch's autograd graph to ensure our custom DAG takes control
            self.data = data.clone().detach()

        self.requires_grad = requires_grad
        self.grad = None  # Stores the gradient (dL/d_self)
        self._op = _op     # Reference to the Operation instance that created this tensor (the "grad_fn" equivalent)
        self._is_leaf = True if requires_grad and _op is None else False

    def zero_grad(self):
        """Sets gradients to None."""
        self.grad = None

    # --- Standard PyTorch-like Tensor Operations (delegating to Operation.apply) ---
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

    # --- The Core Backpropagation Algorithm: DFS-based Reverse Topological Sort ---
    def backward(self, grad_output=None):
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require grad. Set requires_grad=True to enable.")

        # 1. Initialize the root gradient (dL/dL = 1 for the loss tensor itself)
        if grad_output is None:
            grad_output = torch.tensor(1.0, dtype=torch.float32)
        elif not isinstance(grad_output, torch.Tensor):
            grad_output = torch.tensor(grad_output, dtype=torch.float32)

        # 2. Perform DFS to get operations in reverse topological order
        graph_ops_in_reverse_topo_order = []
        visited_nodes = set()

        def _dfs_build_graph(tensor_node):
            if not tensor_node.requires_grad or tensor_node in visited_nodes:
                return

            visited_nodes.add(tensor_node)

            if tensor_node._op is not None:
                # Recursively visit inputs first
                for input_tensor in tensor_node._op.inputs:
                    if isinstance(input_tensor, MyTensor):
                        _dfs_build_graph(input_tensor)
                # Add the operation AFTER all its inputs have been processed
                graph_ops_in_reverse_topo_order.append(tensor_node._op)

        # Start DFS from the current tensor (which is often the loss tensor)
        _dfs_build_graph(self)

        # 3. Initialize gradients for all relevant MyTensors
        gradients = {self: grad_output}

        # 4. Iterate through operations in reverse topological order and propagate gradients
        for op in reversed(graph_ops_in_reverse_topo_order):  # FIXED: Need to reverse the list!
            # Get the accumulated gradient for the output of the current operation
            grad_for_op_output = gradients.get(op.output_tensor)

            if grad_for_op_output is None:
                continue

            # Call the backward method of the operation
            grads_from_op = op.backward(grad_for_op_output)

            # Debug print
            # print(f"Processing {op.__class__.__name__}, got {len(grads_from_op)} gradients")

            # Distribute these gradients back to the inputs of the current operation
            for i, input_tensor in enumerate(op.inputs):
                if isinstance(input_tensor, MyTensor) and input_tensor.requires_grad:
                    current_input_grad = grads_from_op[i]

                    # Accumulate gradients (this is critical for branches/merges)
                    if input_tensor._is_leaf:
                        if input_tensor.grad is None:
                            input_tensor.grad = current_input_grad.clone()
                        else:
                            input_tensor.grad = input_tensor.grad + current_input_grad
                    else:
                        if input_tensor not in gradients:
                            gradients[input_tensor] = current_input_grad.clone()
                        else:
                            gradients[input_tensor] = gradients[input_tensor] + current_input_grad


class _Context:
    """Simple context manager to save tensors/values for the backward pass."""
    def __init__(self):
        self._saved_tensors = ()

    def save_for_backward(self, *tensors):
        # We detach tensors here to ensure they don't carry PyTorch's autograd info
        # into our custom backward computations, preventing double tracking.
        self._saved_tensors = tuple(t.detach() if isinstance(t, torch.Tensor) else t for t in tensors)

    @property
    def saved_tensors(self):
        return self._saved_tensors


class Operation:
    def __init__(self, *inputs):
        # Store references to input MyTensors.
        self.inputs = inputs
        self.output_tensor = None # Will be set by Operation.apply()
        self.ctx = _Context()     # Context to save data needed for backward

    @classmethod
    def apply(cls, *args):
        """
        Static method to create an instance of the operation,
        perform the forward pass, and create the output MyTensor.
        """
        # For most operations, all args are MyTensors
        # For Sum/Mean, args are (my_tensor, dim, keepdim)
        if cls.__name__ in ['Sum', 'Mean']:
            my_tensor = args[0]
            extra_args = args[1:] if len(args) > 1 else ()
            
            if not isinstance(my_tensor, MyTensor):
                my_tensor = MyTensor(my_tensor)
            
            requires_grad = my_tensor.requires_grad
            op_instance = cls(my_tensor, *extra_args)
            
            # Perform forward computation
            output_data = op_instance.forward(my_tensor.data)
            
        else:
            # Regular operations - all args are tensors
            processed_inputs = []
            requires_grad = False
            for inp in args:
                if not isinstance(inp, MyTensor):
                    inp = MyTensor(inp)
                if inp.requires_grad:
                    requires_grad = True
                processed_inputs.append(inp)

            op_instance = cls(*processed_inputs)
            
            # Perform forward computation using the .data attributes of MyTensors
            output_data = op_instance.forward(*[i.data for i in processed_inputs])

        # Create the output MyTensor, linking it back to this operation
        output_tensor = MyTensor(output_data, requires_grad=requires_grad, _op=op_instance)
        op_instance.output_tensor = output_tensor

        return output_tensor

    def forward(self, *input_data):
        """
        Performs the forward computation using torch.Tensors.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Computes gradients with respect to inputs, given grad_output (dL/d_output_of_this_op).
        Must return a tuple of gradients, one for each input.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


# --- Common Operation Subclasses (implementing forward and backward) ---

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
        # For Z = A @ B: dL/dA = dL/dZ @ B.T; dL/dB = A.T @ dL/dZ
        grad_a = torch.matmul(grad_output, b_data.T)
        grad_b = torch.matmul(a_data.T, grad_output)
        return grad_a, grad_b


class Sum(Operation):
    def __init__(self, input_tensor, dim=None, keepdim=False):
        super().__init__(input_tensor)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input_data):
        # Save input shape for correct gradient expansion
        self.ctx.save_for_backward(input_data)
        return input_data.sum(dim=self.dim, keepdim=self.keepdim)

    def backward(self, grad_output):
        input_data, = self.ctx.saved_tensors
        input_shape = input_data.shape

        # Correctly expand grad_output for broadcasting back to original shape
        if self.dim is not None and not self.keepdim:
            # Need to unsqueeze the summed dimensions
            if isinstance(self.dim, int):
                dims_to_unsqueeze = [self.dim]
            else:
                dims_to_unsqueeze = list(self.dim)
            
            for d in sorted(dims_to_unsqueeze):
                grad_output = grad_output.unsqueeze(d)

        # The gradient of sum wrt its input is 1 for each element, broadcasted to input shape
        grad_input = grad_output.expand(input_shape)
        return (grad_input,)


class Mean(Operation):
    def __init__(self, input_tensor, dim=None, keepdim=False):
        super().__init__(input_tensor)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input_data):
        # Calculate num_elements_reduced during forward pass
        if self.dim is not None:
            if isinstance(self.dim, int):
                num_elements_reduced = input_data.shape[self.dim]
            else: # tuple of dims
                num_elements_reduced = 1
                for d in self.dim:
                    num_elements_reduced *= input_data.shape[d]
        else: # Mean over all elements
            num_elements_reduced = input_data.numel()
        
        self.ctx.save_for_backward(input_data, torch.tensor(num_elements_reduced, dtype=torch.float32))
        return input_data.mean(dim=self.dim, keepdim=self.keepdim)

    def backward(self, grad_output):
        input_data, num_elements_reduced_tensor = self.ctx.saved_tensors
        input_shape = input_data.shape
        num_elements_reduced = num_elements_reduced_tensor.item()

        if self.dim is not None and not self.keepdim:
            # Need to unsqueeze the averaged dimensions
            if isinstance(self.dim, int):
                dims_to_unsqueeze = [self.dim]
            else:
                dims_to_unsqueeze = list(self.dim)
            
            for d in sorted(dims_to_unsqueeze):
                grad_output = grad_output.unsqueeze(d)

        # The gradient of mean wrt its input is 1/N for each element
        grad_input = (grad_output / num_elements_reduced).expand(input_shape)
        return (grad_input,)


class ReLU(Operation):
    def forward(self, x_data):
        self.ctx.save_for_backward(x_data)
        return torch.relu(x_data)

    def backward(self, grad_output):
        x_data, = self.ctx.saved_tensors
        # Derivative of ReLU is 1 if x > 0, else 0
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


# Test the fixed implementation
if __name__ == "__main__":
    # Create leaf tensors
    a = MyTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), requires_grad=True)
    b = MyTensor(torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32), requires_grad=True)
    bias = MyTensor(torch.tensor([0.1, 0.2], dtype=torch.float32), requires_grad=True)

    print(f"a:\n{a.data}\n")
    print(f"b:\n{b.data}\n")
    print(f"bias:\n{bias.data}\n")

    # --- Building a more complex DAG (like a simplified two-layer network) ---

    # Layer 1
    linear1_out = a.matmul(b) # Output shape (2,2)
    print(f"Linear1 Output (a @ b):\n{linear1_out.data}\n  _op: {linear1_out._op.__class__.__name__}\n")

    activated1 = linear1_out.relu() # Apply ReLU
    print(f"Activated1 (ReLU):\n{activated1.data}\n  _op: {activated1._op.__class__.__name__}\n")

    # Layer 2 (with residual connection-like behavior for demonstration)
    c = activated1 + a # MERGE: activated1 and a are added. 'a' contributes via two paths!
    print(f"c (activated1 + a):\n{c.data}\n  _op: {c._op.__class__.__name__}\n")

    d = c.sum(dim=1) # Sum across rows, output shape (2,)
    print(f"d (c.sum(dim=1)):\n{d.data}\n  _op: {d._op.__class__.__name__}\n")

    # Final scalar loss
    loss = d.mean() # Mean of the summed values, scalar output
    print(f"Loss (d.mean()):\n{loss.data}\n  _op: {loss._op.__class__.__name__}\n")

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
    bias_torch = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)

    linear1_out_torch = a_torch.matmul(b_torch)
    activated1_torch = F.relu(linear1_out_torch)

    c_torch = activated1_torch + a_torch
    d_torch = c_torch.sum(dim=1)
    loss_torch = d_torch.mean()

    loss_torch.backward()

    print(f"PyTorch's a.grad:\n{a_torch.grad}\n")
    print(f"PyTorch's b.grad:\n{b_torch.grad}\n")
    print(f"PyTorch's bias.grad:\n{bias_torch.grad}\n")
    
    # Debug: Check if gradients match
    print("Gradients match PyTorch:", torch.allclose(a.grad, a_torch.grad, atol=1e-6) if a.grad is not None else False)
    print("B gradients match PyTorch:", torch.allclose(b.grad, b_torch.grad, atol=1e-6) if b.grad is not None else False)
    