package network;

import java.util.List;

public class Neuron {

    private final List<Connection> connections; // List of all dependencies
    private double activation = 0; // the neurons current activation
    private double optimalActivation = 1;


    public Neuron(List<Connection> connections){
        this.connections = connections;
    }

    // updates the neurons current activation based on the connections
    public void updateActivation(){
        double newActivation = 0;
        for (Connection connection : connections){
            newActivation += connection.getWeight() * connection.getSourceNeuron().getActivation();
        }
        if (!connections.isEmpty()) { // if not first layer neuron
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
         for (Connection connection : connections){
             double gradient = ((activation-optimalActivation) * (activation* (1- activation)) * (connection.getSourceNeuron().getActivation()));
             connection.setWeight(connection.getWeight() - 0.1 * gradient);
         }
     }

     public void setOptimalActivation(double optimalActivation){
         this.optimalActivation = optimalActivation;
     }

     public double getOptimalActivation(){
         return optimalActivation;
     }

     public List<Connection> getConnections() {
         return connections;
     }

     private double sigmoid(double x){
         return 1 / (1 + Math.exp(-x));
     }
}