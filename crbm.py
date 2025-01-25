import numpy as np
from scipy.special import expit

class CRBM:
    def __init__(self, n_visible, n_hidden, n_cond):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_cond = n_cond

        # Initialize weights
        self.W = np.random.normal(0, 0.001, size=(n_visible, n_hidden))
        self.U = np.random.normal(0, 0.001, size=(n_cond,    n_hidden))
        self.V = np.random.normal(0, 0.001, size=(n_cond,    n_visible))

        # Initialize biases
        self.b = np.zeros(n_visible)
        self.c = np.zeros(n_hidden)

    def sample_hidden(self, v, cond):
        h_probs = expit(np.dot(v, self.W) + np.dot(cond, self.U) + self.c)
        h_sample = np.random.binomial(1, h_probs)
        return h_sample, h_probs

    def sample_visible(self, h, cond):
        v_mean = np.dot(h, self.W.T) + np.dot(cond, self.V) + self.b
        v_sample = v_mean + np.random.normal(0, 0.001, size=v_mean.shape)
        return v_sample

    def contrastive_divergence(self, v_input, cond, k=1, lr=0.01):
        # Positive phase
        h_sample, _ = self.sample_hidden(v_input, cond)
        positive_grad = v_input.T @ h_sample

        # Negative phase
        v_model = v_input
        h_model = h_sample
        for _ in range(k):
            v_model = self.sample_visible(h_model, cond)
            h_model, _ = self.sample_hidden(v_model, cond)
        
        negative_grad = v_model.T @ h_model

        # Compute gradients
        dW = positive_grad - negative_grad
        dU = cond.T @ (h_sample - h_model)
        dV = cond.T @ (v_input - v_model)
        
        dc = np.mean(h_sample - h_model, axis=0)
        db = np.mean(v_input - v_model, axis=0)

        # Update parameters
        self.W += lr * dW
        self.U += lr * dU
        self.V += lr * dV
        self.b += lr * db
        self.c += lr * dc

    def train(self, v_data, cond_data, n_epochs=100):
        reconstruction_errors = []

        for epoch in range(n_epochs):
            # Modify training loop as needed
            
            self.contrastive_divergence(v_data, cond_data)

            # Compute reconstruction error for the epoch
            v_reconstructed = self.reconstruct(v_data, cond_data)
            mse = np.mean((v_data - v_reconstructed) ** 2)
            reconstruction_errors.append(mse)
            print(f"Epoch {epoch + 1}/{n_epochs}, Reconstruction Error: {mse:.4f}")

        return reconstruction_errors

    def reconstruct(self, v, cond):
        h_sample, _ = self.sample_hidden(v, cond)
        v_recon = self.sample_visible(h_sample, cond)
        return v_recon

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Load and preprocess the dataset
    # data = np.random.randn(4, 1000)
    # data /= np.max(data)

    data = np.sin(np.linspace(-np.pi, np.pi, 1000)).reshape(1, -1)

    test_visible = data[:, 500:]
    test_cond = data[:, :500]

    # Initialize the CRBM
    n_visible = 500
    n_cond = 500
    n_hidden = 1000

    crbm = CRBM(n_visible, n_hidden, n_cond)

    # Train the CRBM
    t = 1
    errors = crbm.train(test_visible, test_cond, n_epochs=t)

    # Visualize reconstruction error over epochs
    plt.plot(errors)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize original vs reconstructed for test samples
    v_reconstructed = crbm.reconstruct(test_visible, test_cond)

    v_reconstructed /= np.max(v_reconstructed)

    plt.plot(test_visible[0, :], label="Original (First Feature)", c='r', alpha=0.7)
    plt.plot(v_reconstructed[0, :], label="Reconstructed (First Feature)",c='b', alpha=0.7)
    plt.legend()
    plt.grid(True)
    plt.show()