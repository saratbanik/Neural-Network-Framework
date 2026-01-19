#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

float mse(const vector<float>& y_true, const vector<float>& y_pred) {
    float loss = 0.0f;
    for (size_t i = 0; i < y_true.size(); ++i) {
        float diff = y_pred[i] - y_true[i];
        loss += diff * diff;
    }
    return loss / y_true.size();
}

class DenseLayer {
public:
    vector<vector<float>> weights;
    vector<float> biases;
    vector<float> last_input;
    vector<float> last_output;
    vector<float> z;

    DenseLayer(int inputSize, int outputSize) {
        weights.resize(outputSize, vector<float>(inputSize));
        biases.resize(outputSize, 0.0f);
        srand((unsigned int)time(0));
        for (auto& row : weights)
            for (auto& w : row)
                w = ((float)rand() / RAND_MAX - 0.5f);  // random init [-0.5, 0.5]
    }

    vector<float> forward(const vector<float>& input) {
        last_input = input;
        z.clear();
        last_output.clear();
        for (size_t i = 0; i < weights.size(); ++i) {
            float sum = biases[i];
            for (size_t j = 0; j < input.size(); ++j) {
                sum += weights[i][j] * input[j];
            }
            z.push_back(sum);
            last_output.push_back(sigmoid(sum));
        }
        return last_output;
    }

    vector<float> backward(const vector<float>& grad_output, float learning_rate) {
        vector<float> grad_input(last_input.size(), 0.0f);
        for (size_t i = 0; i < weights.size(); ++i) {
            float dz = grad_output[i] * sigmoid_derivative(z[i]);
            for (size_t j = 0; j < last_input.size(); ++j) {
                grad_input[j] += weights[i][j] * dz;
                weights[i][j] -= learning_rate * dz * last_input[j];
            }
            biases[i] -= learning_rate * dz;
        }
        return grad_input;
    }
};

int main() {
    DenseLayer layer1(2, 2);
    DenseLayer layer2(2, 1);

    vector<vector<float>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    vector<float> targets = {0, 1, 1, 0};  // XOR

    float lr = 0.1f;
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (int i = 0; i < 4; ++i) {
            vector<float> out1 = layer1.forward(inputs[i]);
            vector<float> out2 = layer2.forward(out1);

            vector<float> loss_grad = {2 * (out2[0] - targets[i])};  // dMSE/dy
            vector<float> grad2 = layer2.backward(loss_grad, lr);
            layer1.backward(grad2, lr);

            total_loss += mse({targets[i]}, out2);
        }
        if (epoch % 1000 == 0)
            cout << "Epoch " << epoch << ", Loss: " << total_loss / 4 << endl;
    }

    cout << "\nPredictions after training:\n";
    for (const auto& x : inputs) {
        vector<float> out = layer2.forward(layer1.forward(x));
        cout << x[0] << " XOR " << x[1] << " = " << out[0] << endl;
    }

    return 0;
}