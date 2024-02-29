import numpy as np
import torch
from utils import make_lexicon, run_rsa

def amortize_outer(matrix):
    # Ensure input is a numpy array for compatibility
    matrix = np.array(matrix)
    
    # create a matrix of 0s of the same size
    mat_bool = np.zeros(matrix.shape)
    # get all the non-zero elements from the input matrix, and put 1s in mat_bool
    mat_bool[matrix != 0] = 1

    # create two vectors that outer-product into a matrix of the same size as the input matrix
    vec_row = np.ones(matrix.shape[0])
    vec_col = np.ones(matrix.shape[1])

    # create a pytorch tensor from mat_bool, vec_row, and vec_col, and matrix
    mat_bool = torch.from_numpy(mat_bool).float()
    vec_row = torch.from_numpy(vec_row).float().requires_grad_(True)
    vec_col = torch.from_numpy(vec_col).float().requires_grad_(True)
    matrix = torch.from_numpy(matrix.astype(np.float32))

    # create a reconstructed matrix of vec_row outerproduct vec_col times mat_bool
    # torch.ger is deprecated in favor of torch.outer in newer versions of PyTorch
    recon_matrix = torch.outer(vec_row, vec_col) * mat_bool
    # the loss function is the L2 of the difference between the reconstructed matrix and the input matrix
    loss = torch.sum((recon_matrix - matrix) ** 2)

    # create an optimizer
    optimizer = torch.optim.Adam([vec_row, vec_col], lr=0.01)
    # optimize the loss function for 1000 iterations
    for i in range(10000):
        optimizer.zero_grad()  # Reset gradients to zero for each iteration
        loss.backward(retain_graph=True)
        optimizer.step()

        # Recompute the loss after updating the variables
        recon_matrix = torch.outer(vec_row, vec_col) * mat_bool
        loss = torch.sum((recon_matrix - matrix) ** 2)

    # Return the outer product of the optimized vec_row and vec_col, ensuring no gradient information is returned
    # also return the boolean matrix and the loss
    return torch.outer(vec_row, vec_col).detach().numpy(), mat_bool.detach().numpy(), loss.item()

if __name__ == '__main__':

    lexicon = make_lexicon(10, 5)
    lexicon, listeners, speakers = run_rsa(lexicon)
    s1 = speakers[1]
    l0 = listeners[0]
    l1 = listeners[1]

    l1_outer, l1_bool, loss_value = amortize_outer(l1)
    l1_amortized = l1_outer * l1_bool
    print (loss_value)

    # # visualize both l1 and l1_amortized as 2D heat map
    import matplotlib.pyplot as plt
    # # put both on the same plot
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(l1, cmap='gray')
    axs[0].set_title('l1')
    axs[1].imshow(l1_amortized, cmap='gray')
    axs[1].set_title('l1_amortized')
    plt.show()
