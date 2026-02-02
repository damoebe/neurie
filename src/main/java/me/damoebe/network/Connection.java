package me.damoebe.network;

import java.util.Random;

/**
 * A class used to connect two neurons by making one the source neuron and the other one the neuron containing the connection
 */
public class Connection {
    /**
     * The weight of the connection
     */
    private double weight;
    /**
     * The source/target neuron
     */
    private final Neuron sourceNeuron;

    /**
     * The main constructor for the Connection class
     * @param targetNeuron The source/target neuron
     */
    public Connection(Neuron targetNeuron) {
        // random weight initialization
        Random random = new Random();
        this.weight = random.nextFloat()*4 - 2;
        this.sourceNeuron = targetNeuron;
    }

    /**
     * Getter for the weight
     * @return the connections weight
     */
    public double getWeight() {
        return weight;
    }

    /**
     * Setter for the weight
     * @param weight The new weight
     */
    public void setWeight(double weight){
        this.weight = weight;
    }

    /**
     * Getter for the source neuron
     * @return The source neuron of the connection
     */
    public Neuron getSourceNeuron() {
        return sourceNeuron;
    }
}