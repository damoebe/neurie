package network;

import java.util.Random;

public class Connection {
    private double weight;
    private final Neuron sourceNeuron;

    public Connection(Neuron targetNeuron) {
        Random random = new Random();
        this.weight = (float) (random.nextInt(20) - 10) /10; // initial weight for network
        this.sourceNeuron = targetNeuron;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight){
        this.weight = weight;
    }

    public Neuron getSourceNeuron() {
        return sourceNeuron;
    }
}