package me.damoebe.network;

import java.util.List;
import java.util.Random;

/**
 * The class of Neurons used in the Networks
 */
public class Neuron{
    /**
     * A list of all connection to neurons in the previous layer
     */
    private final List<Connection> connections;
    /**
     * The neurons current activation
     */
    private double activation = 0;
    /**
     * The neurons bias value
     */
    private double bias = Math.random() * 2 - 1;
    /**
     * The neurons delta used for backpropagation and gradient decent learning
     */
    private double delta;

    /**
     * Main constructor of the Neuron class
     * @param connections An array featuring all dependencies of the neuron as connections
     */
    public Neuron(List<Connection> connections){
        this.connections = connections;
    }

    /**
     * Updates the current activation based on the connections and sigmoid activation function
     */
    public void updateActivation(){
        double newActivation = 0;
        for (Connection connection : connections){
            newActivation += connection.getWeight() * connection.getSourceNeuron().getActivation();
        }
        if (!connections.isEmpty()) { // if not first layer neuron
            newActivation += bias; // add bias to activation calc
            activation = sigmoid(newActivation);
        }
    }

    /**
     * Getter for current Activation
     * @return current activation
     */
    public double getActivation(){
        return activation;
    }

    /**
     * Sets the current activation (used for neurons in the input layer)
     * @param activation The new activation of the neuron
     */
    public void setActivation(double activation){
        this.activation = activation;
    }

    /**
     * Updates all connection weights based on delta value, loss, noise and learningRate
     * @param loss The network loss
     * @param noiseRate The noiseRate for the randomizer
     * @param learningRate The learning Rate which is used to update the weights
     */
    public void updateWeights(double loss, double noiseRate, double learningRate){
        for (Connection connection : connections) {
            // get gradient
            double gradient = delta * connection.getSourceNeuron().getActivation();

            Random random = new Random();
            double noise = (learningRate + (loss*learningRate)) // noise should decent the lower the loss
                    *(random.nextFloat()*2-1)*noiseRate; // actual noise

            connection.setWeight((connection.getWeight() - learningRate * gradient) + noise);
        }
    }

    /**
     * Updates the bias using learningRate and delta
     * @param learningRate The learningRate of the network
     */
    public void updateBias(double learningRate){
        bias = bias - learningRate * delta;
    }

    /**
     * Getter for connections
     * @return List of all connections
     */
    public List<Connection> getConnections() {
        return connections;
    }

    /**
     * Getter for delta
     * @return the neurons delta
     */
    public double getDelta() {
        return delta;
    }

    /**
     * Setter for delta
     * @param delta new delta
     */
    public void setDelta(double delta) {
        this.delta = delta;
    }

    /**
     * Sigmoid function
     * @param x The number that should be turnt into a number from 0-1
     * @return the number from 0-1
     */
    private double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }
}
