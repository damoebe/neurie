package network;

import java.util.Random;

public class Connection {
    private double weight;
    private final Neuron sourceNeuron;

    public Connection(Neuron targetNeuron) {
        Random random = new Random();
        float weight = random.nextFloat(2) - 1;
        if (weight == 0){
            weight = (float) 0.1;
        }
        this.weight =  weight;
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