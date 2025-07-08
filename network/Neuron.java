package network;

import java.util.List;

public class Neuron {

    private final List<Connection> connections; // List of all dependencies
    private final double learningRate;
    private double activation = 0; // the neurons current activation
    private double bias = Math.random() * 2 - 1;
    private double delta; // used for backpropagation and gradient decent learning


    public Neuron(List<Connection> connections, double learningRate){
        this.connections = connections;
        this.learningRate = learningRate;
    }

    // updates the neurons current activation based on the connections
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

    public double getActivation(){
        return activation;
    }

    // for input neurons
    public void setActivation(double activation){
        this.activation = activation;
    }

    public void updateWeights(){
        for (Connection connection : connections) {
            double input = connection.getSourceNeuron().getActivation();
            double gradient = delta * input;
            connection.setWeight(connection.getWeight() - learningRate * gradient);
        }
    }

    public void updateBias(){
        bias = bias - learningRate * delta;
    }
    public List<Connection> getConnections() {
        return connections;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    private double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }
}
