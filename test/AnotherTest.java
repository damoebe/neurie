package test;

import network.Network;
import java.util.List;

public class AnotherTest {
    public static void main(String[] args) {
        // most simple test for backpropagation
        Network network = new Network(1, 1, 1, 1, 0.1);
        int epochs = 10000;
        for (int epoch = 0; epoch != epochs; epoch++){
            network.train(List.of(0.1), List.of(1.0));
            System.out.println(network.getOutput().getFirst() + " Target = 1.0");
        }
    }
}
