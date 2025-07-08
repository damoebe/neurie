package test;

import java.util.Arrays;
import java.util.List;
import network.Network;

public class XORTest {
    public static void main(String[] args) {
        // test code (made by chat-gpt)
        Network network = new Network(2, 1, 4, 1, 0.1);

        // XOR training data
        List<List<Double>> inputs = Arrays.asList(
                Arrays.asList(0.0, 0.0), // 0, 0
                Arrays.asList(0.0, 1.0), // 0, 1
                Arrays.asList(1.0, 0.0), // 1, 0
                Arrays.asList(1.0, 1.0) // 1, 1
        );

        // optimal activations
        List<List<Double>> targets = Arrays.asList(
                List.of(0.0), // 0
                List.of(1.0), // 1
                List.of(1.0), // 1
                List.of(0.0) // 0
        );

        // training
        int epochs = 10000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            for (int i = 0; i < inputs.size(); i++) {
                network.train(inputs.get(i), targets.get(i));
                double prediction = network.getOutput().get(0);
                double error = prediction - targets.get(i).get(0);
                totalLoss += error * error;
            }
            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d, Loss: %.6f%n", epoch, totalLoss / inputs.size());
            }
        }

        System.out.println("\nResults after training:");
        for (int i = 0; i < inputs.size(); i++) {
            network.insertInput(inputs.get(i));
            network.updateAllActivations();
            double output = network.getOutput().get(0);
            System.out.printf("Input: %s → Output: %.4f (expected: %.1f)%n",
                    inputs.get(i).toString(), output, targets.get(i).get(0));
        }
    }
}

