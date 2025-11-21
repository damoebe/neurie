package me.damoebe.network;

import java.util.Random;

public class Connection {
    private double weight;
    private final Neuron sourceNeuron;

    public Connection(Neuron targetNeuron) {
        Random random = new Random();
        float weight = random.nextFloat()*4 - 2;
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