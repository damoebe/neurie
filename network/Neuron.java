package network;

import java.util.List;

public class Neuron {

    private final List<Connection> connections; // List of all dependencies
    private double activation = 0; // the neurons current activation


    public Neuron(List<Connection> connections){
        this.connections = connections;
    }

    // updates the neurons current activation based on the connections
    public void updateActivation(){
        double newActivation = 0;
        for (Connection connection : connections){
            newActivation += connection.getWeight() * connection.getTargetNeuron().getActivation();
        }
        if (!connections.isEmpty()) { // if not first layer neuron
            activation = newActivation / connections.size();
        }
    }

    public double getActivation(){
        return activation;
    }

    // for input neurons
    public void setActivation(double activation){
        this.activation = activation;
    }
}
